from PIL import Image, ImageDraw, ImageOps
import cv2
import numpy as np
from numpy.typing import NDArray
import math
import cProfile
from dataclasses import dataclass
from tesserocr import PyTessBaseAPI, PSM
import matplotlib.pyplot as plt
import win32gui
import win32ui
import win32con
import time
from difflib import SequenceMatcher
import re
from enum import Enum

KNOWN_H = 408
KNOWN_W = 563
KNOWN_CENTER_X = 282
KNOWN_CENTER_Y = 184


Gamestate = Enum('Gamestate', ['UNKNOWN', 'PRE_FLOP', 'FLOP', 'TURN', 'RIVER', 'ALL_IN'])
Playerstate = Enum('Playerstate', ['WAITING', 'CHECKED', 'CALLED', 'RAISED', 'BET', 'ALL_IN', 'FOLDED', 'SITTING_OUT'])
ANGLES = {
    6: [0.73, 1.93, 2.39, 3.21, 4.69, 5.34],
    9: [0.98, 1.75, 2.09, 2.38, 2.75, 4.09, 4.69, 5.08, 5.61]
}


@dataclass
class Position:
    x: int
    y: int


@dataclass
class PlayerMatch(Position):
    confidence: float
    leftSide: bool


@dataclass
class Template:
    w: int
    h: int
    template: NDArray
    mask: NDArray


@dataclass
class Player:
    match: PlayerMatch
    names: list[str]
    state: Playerstate
    prev_name: str
    hands: int
    made_action: bool
    vpip: int
    pfr: int
    stack: float
    special: bool
    unknown_count: int


@dataclass
class Game:
    state: Gamestate
    players: list[Player]
    player_locs: list[Position]
    utg: int


def get_name(player: Player):
    return player.names[0] if len(player.names) > 0 else '???'


def fit_ellipse(image: NDArray, global_best_ratio: float):
    TARGET_AREA = 54662

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 143, 27), (255, 255, 180))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    best_contour = -1
    max_area = -1
    for i in range(len(contours)):
        new_area = cv2.contourArea(contours[i])
        if abs(TARGET_AREA - (new_area / (global_best_ratio**2))) < 5000:
            best_contour = i

    if best_contour == -1:
        print('error detecting table, defaulting')
        guess_center = (KNOWN_CENTER_X * global_best_ratio, KNOWN_CENTER_Y * global_best_ratio)
        return guess_center

    ellipse = cv2.fitEllipse(contours[best_contour])

    return (int(ellipse[0][0]), int(ellipse[0][1]))


def name_closeness(name, known_names):
    if len(known_names) == 0:
        return 0
    return sum(similarity(name, known) for known in known_names) / len(known_names)


def similarity(a: str, b: str):
    return SequenceMatcher(None, a, b).ratio()


def dist(p: Position, q: Position):
    return math.hypot(p.x - q.x, p.y - q.y)


def is_name_special(name):
    special_names = ['Bet', 'Call', 'Check', 'Fold', 'Muck', 'Post BB', 'Post SB', 'Post SB&BB', 'Raise', 'Show Cards']
    return name in special_names or name.startswith('Won')


def sparse_subset(points, r):
    'Create groups of matches that are approximately within r distance of each other'
    result = []
    for p in points:
        is_neighbor = False
        neighborhood = []
        for q, nearby in result:
            if dist(p, q) <= r:
                is_neighbor = True
                neighborhood = nearby
        if is_neighbor:
            neighborhood.append(p)
        else:
            result.append((p, [p]))
    return result


def find_max_confidence(points):
    'Find the match with the highest confidence'
    best_point = points[0]

    for point in points[1:]:
        if best_point.confidence < point.confidence:
            best_point = point

    return best_point


def load_template() -> Template:
    template_img = cv2.imread('assets/template.png')
    (h, w) = template_img.shape[:2]
    template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    template_mask = cv2.imread('assets/mask.png')
    template_mask = cv2.cvtColor(template_mask, cv2.COLOR_BGR2GRAY)
    _, template_mask = cv2.threshold(template_mask, 127, 255, cv2.THRESH_BINARY)

    return Template(w, h, template_img, template_mask)


def is_playing(player: Player, image: NDArray, template: Template, global_best_ratio: float):
    '''Determine if a player is out of this hand by counting bright green pixels in player's '''
    playermatch = player.match

    x1 = playermatch.x
    y1 = playermatch.y
    x2 = playermatch.x + int(template.w * global_best_ratio)
    y2 = playermatch.y + int(template.h * global_best_ratio)
    section = image[y1:y2, x1:x2]

    scale = 250 / section.shape[0]
    scaled_section = cv2.resize(section, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if playermatch.leftSide:
        avatarCoords = (656, 125)
    else:
        avatarCoords = (-62, 125)
    scaled_section = cv2.circle(scaled_section, avatarCoords, 132, (0, 0, 0), -1)

    stack_section = scaled_section[130:232, 100:495]
    stack_section = cv2.cvtColor(stack_section, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(stack_section, (49, 40, 147), (69, 80, 255))

    return np.sum(mask == 255) > 45


def find_players(image: NDArray, template: Template) -> tuple[list[PlayerMatch], float]:
    THRESH = 0.70

    h = image.shape[0]

    small_ratio = KNOWN_H * 0.8 / h
    large_ratio = KNOWN_H * 1.2 / h

    # print(f"{small_ratio:.2}:{large_ratio:.2}")

    locations = []
    global_max_val = 0
    global_best_ratio = 0
    for scale in np.linspace(small_ratio, large_ratio, 14)[::-1]:

        resized = cv2.resize(image, dsize=None, fx=scale, fy=scale)
        (resize_h, resize_w) = resized.shape[:2]
        resized_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        test_ratio = image.shape[1] / float(resized.shape[1])

        if resized.shape[0] < template.h or resized.shape[1] < template.w:
            break

        results = []

        # search right side
        x_offset = int(resize_w * .35)
        result = cv2.matchTemplate(resized_gray[:, x_offset:], template.template,
                                   cv2.TM_CCOEFF_NORMED, mask=template.mask)

        (_, max_val, _, max_loc) = cv2.minMaxLoc(result)
        max_val_right = max_val
        while max_val > THRESH:
            if max_val < 1:
                results.append(PlayerMatch(max_loc[0] + x_offset, max_loc[1], max_val, False))
            result = cv2.circle(result, max_loc[:2], int(80 * scale), (0, 0, 0), -1)
            (_, max_val, _, max_loc) = cv2.minMaxLoc(result)

        # search left side
        result = cv2.matchTemplate(resized_gray[:, :int(resize_w*.65)], cv2.flip(template.template, 1),
                                   cv2.TM_CCOEFF_NORMED, mask=cv2.flip(template.mask, 1))

        (_, max_val, _, max_loc) = cv2.minMaxLoc(result)
        max_val_left = max_val
        while max_val > THRESH:
            if max_val < 1:
                results.append(PlayerMatch(max_loc[0], max_loc[1], max_val, True))
            result = cv2.circle(result, max_loc[:2], int(80 * scale), (0, 0, 0), -1)
            (_, max_val, _, max_loc) = cv2.minMaxLoc(result)

        # Convert match coordinates to standard size
        for match in results:
            match.x = int(match.x * test_ratio)
            match.y = int(match.y * test_ratio)
            locations.append(match)

        # If confidence is decreasing appreciably, end
        max_val = (max_val_left + max_val_right) / 2
        if global_max_val - max_val > .1:
            break

        if global_max_val < max_val:
            global_max_val = max_val
            global_best_ratio = test_ratio

    filtered_locations = sparse_subset(locations, int(h * .14))
    players = [find_max_confidence(close_points) for _, close_points in filtered_locations]

    print(f"ratio {global_best_ratio} (guess {h / KNOWN_H})")
    return players, global_best_ratio


def fast_find_players(game: Game,
                      image: NDArray,
                      template: Template,
                      global_best_ratio: float):
    for player_loc in game.player_locs:
        #TODO: finish this
        pass


def mod_distance(a, b, n):
    diff = abs((a % n) - (b % n))
    return min(diff, n - diff)


def position_error(players: list[PlayerMatch],
                   angles: list[float],
                   center_x: float,
                   center_y: float):
    error = 0
    for player in players:
        raw_angle = math.atan2(player.y - center_y, player.x - center_x)
        player_angle = ((raw_angle) + (3.65*math.pi)) % (2*math.pi)

        i = 0
        while i < len(angles) and player_angle > angles[i]:
            i += 1
        
        i %= len(angles)
        
        error += min(mod_distance(player_angle, angles[i], 2*math.pi),
                     mod_distance(player_angle, angles[(i - 1) % len(angles)], 2*math.pi))
    error /= len(players)
    print(f"error({len(angles)}): {error}")
    return error



def add_players(game: Game,
                players: list[PlayerMatch],
                all_players: list[Player],
                image: NDArray,
                template: Template,
                global_best_ratio: float,
                api):
    game.ellipse_match = fit_ellipse(image, global_best_ratio)
    center_x, center_y = game.ellipse_match

    known_x, known_y = 405, 308
    
    if len(players) > 6:
        test_error = position_error(players, ANGLES[9], center_x, center_y)
        if test_error > .3:
            print('FATAL: UNKNOWN PLAYER COUNT')
            exit()
        num_players = 9
    else:
        num_players = 0
        error = 1e9

        for test_num, test_angles in ANGLES.items():
            test_error = position_error(players, test_angles, center_x, center_y)
            if test_error < error:
                num_players = test_num
                error = test_error
    
    print(num_players)



    player_angles = []
    game.players.clear()

    for player in players:
        name, stack, valid = parse_player(player, image, template, global_best_ratio, api, False)
        # print(f" {name}:{stack}")
        if valid:
            names = []
            special = is_name_special(name)

            if not special:
                names.append(name)

            if stack == -1:
                state = Playerstate.SITTING_OUT
            else:
                state = Playerstate.FOLDED

            # Find angle around table
            raw_angle = math.atan2(player.y - center_y, player.x - center_x)
            converted_angle = ((raw_angle) + (3.65*math.pi)) % (2*math.pi)

            i = 0
            while i < len(player_angles) and player_angles[i] < converted_angle:
                i += 1

            player_angles.insert(i, converted_angle)

            player_obj = None
            if len(names) > 0:
                for known_player in all_players:
                    if name_closeness(names[0], known_player.names) > 0.7:
                        print(f'found player {known_player.names[0]} with {known_player.hands} hands')
                        player_obj = known_player

            if player_obj is None:
                player_obj = Player(match=player,
                                    names=names,
                                    state=state,
                                    prev_name='',
                                    hands=0,
                                    made_action=False,
                                    vpip=0,
                                    pfr=0,
                                    stack=stack,
                                    special=special,
                                    unknown_count=0)
                all_players.append(player_obj)
            game.players.insert(i, player_obj)

            player_pos = Position(player.x, player.y)
            close_pos = False
            for other_pos in game.player_locs:
                if dist(player_pos, other_pos) < 50:
                    close_pos = True

            if not close_pos:
                game.player_locs.insert(i, player_pos)

    for i, (player, angle) in enumerate(zip(game.players, player_angles)):
        print(f" {i}: {angle:.2} ({player.names[0] if len(player.names) > 0 else '.'})")


def parse_player(playermatch: PlayerMatch,
                 image: NDArray,
                 template: Template,
                 global_best_ratio: float,
                 api: PyTessBaseAPI,
                 debug: bool) -> tuple[str, float, bool]:
    '''Parses a PlayerMatch to retrieve the player's name/action and stack. Additionally returns
    a bool to indicate if the parsing failed.
    '''

    x1 = playermatch.x
    y1 = playermatch.y
    x2 = playermatch.x + int(template.w * global_best_ratio)
    y2 = playermatch.y + int(template.h * global_best_ratio)
    section = image[y1:y2, x1:x2]

    scale = 250 / section.shape[0]
    scaled_section = cv2.resize(section, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if playermatch.leftSide:
        avatarCoords = (656, 125)
    else:
        avatarCoords = (-62, 125)
    scaled_section = cv2.circle(scaled_section, avatarCoords, 132, (0, 0, 0), -1)

    # Isolate the player name
    name_section = scaled_section[35:120, 65:530]
    # cv2.imshow('toptext', name_section)
    name_section = cv2.cvtColor(name_section, cv2.COLOR_BGR2HSV)
    mask_name = cv2.inRange(name_section, (0, 0, 53), (255, 25, 255))
    mask_blue = cv2.inRange(name_section, (97, 88, 29), (102, 174, 255))

    blue_pixels = np.sum(mask_blue == 255)
    if blue_pixels > 30:
        name_section = cv2.bitwise_and(name_section, name_section, mask=mask_blue)
    else:
        name_section = cv2.bitwise_and(name_section, name_section, mask=mask_name)
    name_section = cv2.cvtColor(name_section, cv2.COLOR_HSV2BGR)

    if debug:
        cv2.imshow('debug', name_section)
        cv2.imshow('blue', mask_blue)
        cv2.imshow('name', mask_name)

    api.SetImage(Image.fromarray(name_section))
    name = api.GetUTF8Text()
    name = name.strip()
    # Remove likely errors in name
    while len(name) > 0 and not name[0].isalpha():
        name = name[1:]
    while len(name) > 0 and name[-1] in ' \\/,.`\'"':
        name = name[:-1]

    if debug:
        print(f'*** {blue_pixels > 30} >{name}< ***')
        cv2.waitKey(0)

    # Isolate the player chip stack
    stack_section = scaled_section[130:232, 100:495]
    # cv2.imshow('bottomtext', stack_section)
    stack_section = cv2.cvtColor(stack_section, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(stack_section, (44, 0, 0), (72, 255, 255))
    stack_section = cv2.bitwise_and(stack_section, stack_section, mask=mask)
    stack_section = cv2.cvtColor(stack_section, cv2.COLOR_HSV2BGR)

    api.SetImage(Image.fromarray(stack_section))
    line = api.GetUTF8Text()
    line = line.strip()

    if re.search(r'sitting ?out', line.lower()) is not None:
        stack_val = -1.0
    elif re.search(r'all ?in', line.lower()) is not None:
        stack_val = 0.0
    else:
        match = re.search(r'([\d]+(\.\d)?) ?BB', line)
        if match is None:
            return '', 0, False
        else:
            stack_val = float(match.group(1))

    return name, stack_val, True


def update_player(game: Game,
                  player: Player,
                  all_players: list[Player],
                  position: int,
                  image: NDArray,
                  template: Template,
                  global_best_ratio: float,
                  api):
    name, stack, valid = parse_player(player.match, image, template, global_best_ratio, api, False)

    if not valid:
        # must_search = True
        player.unknown_count += 1
        if player.unknown_count > 4:
            print(f">>invalid (expect {player.names[:3]})")
            return True, False
        else:
            return False, False
    if stack == -1.0:
        player.state = Playerstate.SITTING_OUT
        return False, False
    else:
        if player.state == Playerstate.SITTING_OUT:
            player.state = Playerstate.FOLDED

        player.special = False
        new_hand = False
        if is_name_special(name):
            if player.prev_name != name:
                if name in ['Post BB', 'Post SB&BB']:
                    new_hand = True
                    game.utg = (position + 1) % 6
                elif name == 'Post SB':

                    # game.utg = (position + 2) % 6
                    pass
                elif name == 'Fold':
                    player.state = Playerstate.FOLDED
                elif name in ['Bet', 'Raise']:
                    for to_update in game.players:
                        if to_update.state in [Playerstate.CHECKED,
                                               Playerstate.CALLED,
                                               Playerstate.RAISED,
                                               Playerstate.BET]:
                            to_update.state = Playerstate.WAITING
                    if stack > 0:
                        player.state = Playerstate.BET if name == 'Bet' else Playerstate.RAISED
                    else:
                        player.state = Playerstate.ALL_IN
                    if game.state is Gamestate.PRE_FLOP and not player.made_action:
                        player.vpip += 1
                        player.pfr += 1
                        player.made_action = True
                elif name == 'Check':
                    player.state = Playerstate.CHECKED
                elif name == 'Call':
                    player.state = Playerstate.CALLED
                    if game.state is Gamestate.PRE_FLOP and not player.made_action:
                        player.vpip += 1
                        player.made_action = True

            player.special = True
        elif not name in player.names:
            if len(player.names) == 0:
                for known_player in all_players:
                    if name_closeness(name, known_player.names) > 0.7:
                        game.players[position] = known_player

                player.names.append(name)
                player.unknown_count = 0
            else:
                closeness = sum(similarity(name, known) for known in player.names) / len(player.names)
                if closeness > .7:
                    player.names.append(name)
                    player.unknown_count = 0
                else:
                    player.unknown_count += 1
                    if player.unknown_count > 4:
                        print(f">>triggered {name} (expect {player.names[:3]})")
                        return True, False
                        # parse_player(player.match, image, template, global_best_ratio, api, True)
        else:
            player.unknown_count = 0
            playing = is_playing(player, image, template, global_best_ratio)
            if not playing and not player.state is Playerstate.FOLDED:
                print(f" {get_name(player)} from {player.state} to FOLDED")
                player.state = Playerstate.FOLDED

        player.stack = stack
        player.prev_name = name

        return False, new_hand


def update_game_state(game: Game):
    if (game.state is Gamestate.PRE_FLOP or
        game.state is Gamestate.FLOP or
            game.state is Gamestate.TURN):

        ready = 0
        all_in = 0
        folded = 0
        out = 0
        for player in game.players:
            if player.state is Playerstate.ALL_IN:
                all_in += 1
            elif player.state is Playerstate.FOLDED:
                folded += 1
            elif player.state is Playerstate.SITTING_OUT:
                out += 1
            elif not player.state is Playerstate.WAITING:
                ready += 1

        if ready + all_in + folded + out == len(game.players):
            if ready < 2 and all_in > 0:
                game.state = Gamestate.ALL_IN
            else:
                if game.state == Gamestate.PRE_FLOP:
                    game.state = Gamestate.FLOP
                elif game.state == Gamestate.FLOP:
                    game.state = Gamestate.TURN
                elif game.state == Gamestate.TURN:
                    game.state = Gamestate.RIVER
                for player in game.players:
                    if not player.state in [Playerstate.SITTING_OUT, Playerstate.FOLDED]:
                        player.state = Playerstate.WAITING


def draw_players(game: Game, image: NDArray, template: NDArray, global_best_ratio: float):
    gamestate = game.state.name.replace('_', ' ')
    cv2.putText(image, gamestate, (96, 64), cv2.FONT_HERSHEY_SIMPLEX, .8, (204, 204, 224), 1)

    center_x, center_y = game.ellipse_match
    cv2.circle(image, (int(center_x), int(center_y)), 4, (255, 128, 0), -1)

    for i, player in enumerate(game.players):
        match = player.match
        x1 = match.x
        y1 = match.y
        x2 = match.x + int(template.w * global_best_ratio)
        y2 = match.y + int(template.h * global_best_ratio)
        rect_color = (64, 255, 64) if player.special else (0, 0, 255)
        cv2.rectangle(image, (x1, y1),  (x2, y2), rect_color, 2)
        if player.state is Playerstate.FOLDED:
            cv2.line(image, (x2, y1), (x1, y2), rect_color, 1)
        if i == game.utg:
            cv2.line(image, (x1, y1), ((x1+x2)//2, (3*y1-y2)//2), rect_color, 1)
            cv2.line(image, (x2, y1), ((x1+x2)//2, (3*y1-y2)//2), rect_color, 1)
            # cv2.circle(image, ((x1+x2)//2, (y1+y2)//2), 70, (64, 255, 64), 1)

        # confidence = f"{match.confidence:.2}"
        # cv2.putText(image, confidence, (x1, y1 - 15),
        #             cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3)
        # cv2.putText(image, confidence, (x1, y1 - 15),
        #             cv2.FONT_HERSHEY_SIMPLEX, .5, (128, 192, 255), 1)
        
        # name = get_name(player)
        # cv2.putText(image, name, (x1 + 50, y1 - 15),
        #             cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3)
        # cv2.putText(image, name, (x1 + 50, y1 - 15),
        #             cv2.FONT_HERSHEY_SIMPLEX, .5, (128, 192, 255), 1)

        # stack = f"{player.stack}"
        # cv2.putText(image, stack, (x1, y1 + 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        # cv2.putText(image, stack, (x1, y1 + 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 192, 255), 1)

        vpip = 0.0 if player.hands == 0 else player.vpip / player.hands
        vpip_str = f"VPIP {player.vpip} / {player.hands} ({vpip:.2})"
        cv2.putText(image, vpip_str, (x1, y1 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(image, vpip_str, (x1, y1 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 192, 255), 1)

        pfr = 0.0 if player.hands == 0 else player.pfr / player.hands
        pfr_str = f"PFR {player.pfr} / {player.hands} ({pfr:.2})"
        cv2.putText(image, pfr_str, (x1, y1+100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(image, pfr_str, (x1, y1+100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 192, 255), 1)

        state_str = player.state.name.replace('_', ' ')
        cv2.putText(image, state_str, (x1, y1 + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(image, state_str, (x1, y1 + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 192, 255), 1)


def main():
    template = load_template()

    api = PyTessBaseAPI(init=False)
    api.InitFull(path=r'C:\Program Files\Tesseract-OCR\tessdata',
                 variables={"load_system_dawg": "F", "load__freq_dawg": "F"})
    api.SetPageSegMode(PSM.SINGLE_LINE)

    windows = []

    def get_windows_handler(hwnd, ctx):
        if win32gui.IsWindowVisible(hwnd):
            name = win32gui.GetWindowText(hwnd)
            if len(name) > 0:
                ctx.append((hwnd, name))
    win32gui.EnumWindows(get_windows_handler, windows)

    for i, (_, name) in enumerate(windows):
        print(f"{i:2}: {name}")

    window_idx = input('Enter index: ')
    hwind = windows[int(window_idx)][0]
    print(hwind)
    rect = win32gui.GetWindowRect(hwind)
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]
    print(f"w:{w} h:{h}")
    wdc = win32gui.GetWindowDC(hwind)
    dc_obj = win32ui.CreateDCFromHandle(wdc)
    c_dc = dc_obj.CreateCompatibleDC()
    data_bitmap = win32ui.CreateBitmap(c_dc)
    data_bitmap.CreateCompatibleBitmap(dc_obj, w, h)

    must_search = True
    game = Game(state=Gamestate.UNKNOWN,
                players=[],
                player_locs=[],
                utg=-1)
    all_players = []

    while True:
        c_dc.SelectObject(data_bitmap)
        c_dc.BitBlt((0, 0), (w, h), dc_obj, (0, 0), win32con.SRCCOPY)

        signed_array = data_bitmap.GetBitmapBits(True)
        image = np.frombuffer(signed_array, dtype='uint8')
        image.shape = (h, w, 4)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        # cv2.imshow('test', image)
        # cv2.waitKey(0)

        if must_search:
            # full search, TODO: implement optimized search:
            player_match_list, global_best_ratio = find_players(image, template)
            if len(player_match_list) == 0:
                continue

            add_players(game,
                        player_match_list,
                        all_players,
                        image,
                        template,
                        global_best_ratio,
                        api)

            if len(game.players) > 1:
                must_search = False

        else:
            new_hand = False
            for i, player in enumerate(game.players):
                must_search_vote, new_hand_vote = update_player(game,
                                                                player,
                                                                all_players,
                                                                i,
                                                                image,
                                                                template,
                                                                global_best_ratio,
                                                                api)

                must_search |= must_search_vote
                new_hand |= new_hand_vote

            if new_hand:
                game.state = Gamestate.PRE_FLOP
                for player in game.players:
                    if not player.state is Playerstate.SITTING_OUT:
                        player.state = Playerstate.WAITING
                        player.made_action = False
                        player.hands += 1
                        displayname = get_name(player)
                        print(f"inc {displayname} hands to {player.hands}")
                print()

            # Update game state
            update_game_state(game)

        draw_players(game, image, template, global_best_ratio)

        cv2.imshow('game', image)
        val = cv2.waitKey(1)
        if val == ord('q'):
            break
        elif val == ord('r'):
            must_search = True
        # cv2.waitKey(0)

    dc_obj.DeleteDC()
    c_dc.DeleteDC()
    win32gui.ReleaseDC(hwind, wdc)
    win32gui.DeleteObject(data_bitmap.GetHandle())


if __name__ == '__main__':
    main()
