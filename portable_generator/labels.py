from pathlib import Path
import re

try:
    import yaml
except ModuleNotFoundError:  # The common YOLO ``names`` layouts need no dependency.
    yaml = None

unit_list = [
    "archer-evolution",
    "archer-queen",
    "archer",
    "arrows",
    "axe",
    "baby-dragon",
    "background-items",
    "backgrounds",
    "balloon",
    "bandit",
    "bar-level",
    "bar",
    "barbarian-barrel",
    "barbarian-evolution",
    "barbarian-hut",
    "barbarian",
    "bat-evolution",
    "bat",
    "battle-healer",
    "battle-ram-evolution",
    "battle-ram",
    "big-text",
    "bomb-tower",
    "bomb",
    "bomber-evolution",
    "bomber",
    "bowler",
    "cannon-cart",
    "cannon",
    "cannoneer-tower",
    "clock",
    "clone",
    "dagger-duchess-tower-bar",
    "dagger-duchess-tower",
    "dark-prince",
    "dart-goblin",
    "dirt",
    "earthquake",
    "electro-dragon",
    "electro-giant",
    "electro-spirit",
    "electro-wizard",
    "elite-barbarian",
    "elixir-collector",
    "elixir-golem-big",
    "elixir-golem-mid",
    "elixir-golem-small",
    "elixir",
    "emote",
    "evolution-symbol",
    "executioner",
    "fire-spirit",
    "fireball",
    "firecracker-evolution",
    "firecracker",
    "fisherman",
    "flying-machine",
    "freeze",
    "furnace",
    "giant-skeleton",
    "giant-snowball",
    "giant",
    "goblin-ball",
    "goblin-barrel",
    "goblin-brawler",
    "goblin-cage",
    "goblin-drill",
    "goblin-giant",
    "goblin-hut",
    "goblin",
    "golden-knight",
    "golem",
    "golemite",
    "graveyard",
    "guard",
    "heal-spirit",
    "hog-rider",
    "hog",
    "hunter",
    "ice-golem",
    "ice-spirit-evolution-symbol",
    "ice-spirit-evolution",
    "ice-spirit",
    "ice-wizard",
    "inferno-dragon",
    "inferno-tower",
    "king-tower-bar",
    "king-tower",
    "knight-evolution",
    "knight",
    "lava-hound",
    "lava-pup",
    "lightning",
    "little-prince",
    "lumberjack",
    "magic-archer",
    "mega-knight",
    "mega-minion",
    "mighty-miner",
    "miner",
    "mini-pekka",
    "minion",
    "monk",
    "mortar-evolution",
    "mortar",
    "mother-witch",
    "musketeer",
    "night-witch",
    "pekka",
    "phoenix-big",
    "phoenix-egg",
    "phoenix-small",
    "poison",
    "prince",
    "princess",
    "queen-tower",
    "rage",
    "ram-rider",
    "rascal-boy",
    "rascal-girl",
    "rocket",
    "royal-delivery",
    "royal-ghost",
    "royal-giant-evolution",
    "royal-giant",
    "royal-guardian",
    "royal-hog",
    "royal-recruit-evolution",
    "royal-recruit",
    "skeleton-barrel",
    "skeleton-dragon",
    "skeleton-evolution",
    "skeleton-king-bar",
    "skeleton-king-skill",
    "skeleton-king",
    "skeleton",
    "small-text",
    "sparky",
    "spear-goblin",
    "tesla-evolution",
    "tesla",
    "the-log",
    "tombstone",
    "tornado",
    "tower-bar",
    "valkyrie-evolution",
    "valkyrie",
    "wall-breaker-evolution",
    "wall-breaker",
    "witch",
    "wizard",
    "x-bow",
    "zap",
    "zappy",
    "royal-chef-tower",
    "musketeer-evolution",
    "goblin-barrel-evolution",
    "cannon-evolution",
    "royal-hog-evolution",
    "dart-goblin-evolution",
    "skeleton-barrel-evolution",
    "baby-dragon-evolution",
    "goblin-cage-evolution",
    "zap-evolution",
    "goblin-drill-evolution",
    "hunter-evolution",
    "electro-dragon-evolution",
    "wizard-evolution",
    "executioner-evolution",
    "skeleton-army-evolution",
    "goblin-giant-evolution",
    "furnace-evolution",
    "mega-knight-evolution",
    "witch-evolution",
    "pekka-evolution",
    "boss-bandit",
    "royal-ghost-evolution",
    "lumberjack-evolution",
    "vines",
    "spirit-empress",
    "goblin-machine",
    "rune-giant",
    "inferno-dragon-evolution",
    "goblinstein",
    "goblin-curse",
    "goblin-demolisher",
    "three-musketeers",
    "minion-horde",
    "suspicious-bush",
    "berserker",
    "giant-snowball-evolution",
    "skeleton-army",
    "mirror",
    "tesla-evolution-shock"
]

def load_class_names(path: str | Path) -> dict[int, str]:
    """Load the ``names`` section from an Ultralytics/YOLO YAML file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text) or {}
        names = data.get("names", {})
        if isinstance(names, list):
            return {idx: str(name) for idx, name in enumerate(names)}
        if isinstance(names, dict):
            return {int(idx): str(name) for idx, name in names.items()}

    # Dependency-free fallback for the two usual YOLO layouts:
    # ``names: [cat, dog]`` and an indented ``0: cat`` mapping.
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if re.match(r"^\s*names\s*:", line)), None)
    if start is None:
        raise ValueError(f"Missing 'names' section in {path}")
    head = lines[start].split(":", 1)[1].strip()
    clean = lambda value: value.split("#", 1)[0].strip().strip("'\"")
    if head.startswith("[") and head.endswith("]"):
        values = [clean(value) for value in head[1:-1].split(",")]
        return {idx: value for idx, value in enumerate(values) if value}

    result: dict[int, str] = {}
    list_values: list[str] = []
    base_indent = len(lines[start]) - len(lines[start].lstrip())
    for line in lines[start + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= base_indent:
            break
        mapping = re.match(r"^\s*(\d+)\s*:\s*(.*?)\s*$", line)
        if mapping:
            result[int(mapping.group(1))] = clean(mapping.group(2))
            continue
        item = re.match(r"^\s*-\s*(.*?)\s*$", line)
        if item:
            list_values.append(clean(item.group(1)))
    if result:
        return result
    if list_values:
        return dict(enumerate(list_values))
    raise ValueError(f"Expected 'names' to be a list or mapping in {path}")


idx2unit: dict[int, str] = {}
unit2idx: dict[str, int] = {}


def configure_class_names(path: str | Path) -> None:
    """Replace the active class mapping without invalidating imported dicts."""
    loaded = load_class_names(path)
    idx2unit.clear()
    idx2unit.update(loaded)
    unit2idx.clear()
    unit2idx.update({name: idx for idx, name in loaded.items()})


_default_classes_path = Path(__file__).with_name("classes.yaml")
if _default_classes_path.exists():
    configure_class_names(_default_classes_path)
ground_unit_list = [
    "archer",
    "archer-evolution",
    "archer-queen",
    "balloon",
    "bandit",
    "barbarian",
    "barbarian-evolution",
    "barbarian-barrel",
    "barbarian-hut",
    "battle-healer",
    "battle-ram",
    "battle-ram-evolution",
    "bomb",
    "bomb-tower",
    "bomber",
    "bomber-evolution",
    "bowler",
    "cannon",
    "cannon-cart",
    "clone",
    "dark-prince",
    "dart-goblin",
    "dirt",
    "earthquake",
    "electro-dragon",
    "electro-giant",
    "electro-spirit",
    "electro-wizard",
    "elite-barbarian",
    "elixir-collector",
    "elixir-golem-big",
    "elixir-golem-mid",
    "elixir-golem-small",
    "executioner",
    "fire-spirit",
    "firecracker",
    "firecracker-evolution",
    "fisherman",
    "freeze",
    "furnace",
    "giant",
    "giant-skeleton",
    "goblin",
    "goblin-brawler",
    "goblin-cage",
    "goblin-drill",
    "goblin-giant",
    "goblin-hut",
    "golden-knight",
    "golem",
    "golemite",
    "graveyard",
    "guard",
    "heal-spirit",
    "hog",
    "hog-rider",
    "hunter",
    "ice-golem",
    "ice-spirit",
    "ice-spirit-evolution",
    "ice-wizard",
    "inferno-tower",
    "knight",
    "knight-evolution",
    "lava-hound",
    "lava-pup",
    "little-prince",
    "lumberjack",
    "magic-archer",
    "mega-knight",
    "mighty-miner",
    "miner",
    "mini-pekka",
    "monk",
    "mortar",
    "mortar-evolution",
    "mother-witch",
    "musketeer",
    "night-witch",
    "pekka",
    "phoenix-egg",
    "poison",
    "prince",
    "princess",
    "rage",
    "ram-rider",
    "rascal-boy",
    "rascal-girl",
    "royal-delivery",
    "royal-ghost",
    "royal-giant",
    "royal-giant-evolution",
    "royal-guardian",
    "royal-hog",
    "royal-recruit",
    "royal-recruit-evolution",
    "skeleton",
    "skeleton-evolution",
    "skeleton-king",
    "skeleton-king-skill",
    "sparky",
    "spear-goblin",
    "tesla",
    "tesla-evolution",
    "tesla-evolution-shock",
    "tombstone",
    "valkyrie",
    "valkyrie-evolution",
    "wall-breaker",
    "wall-breaker-evolution",
    "witch",
    "wizard",
    "wizard-evolution",
    "the-log",
    "x-bow",
    "zappy",
    "phoenix-egg",
]

tower_unit_list = [
    "king-tower",
    "queen-tower",
    "cannoneer-tower",
    "dagger-duchess-tower",
    "royal-chef-tower",
]

flying_unit_list = [
    "arrows",
    "axe",
    "baby-dragon",
    "bat",
    "bat-evolution",
    "flying-machine",
    "fireball",
    "giant-snowball",
    "goblin-ball",
    "goblin-barrel",
    "inferno-dragon",
    "lightning",
    "mega-minion",
    "minion",
    "phoenix-big",
    "phoenix-small",
    "rocket",
    "skeleton-barrel",
    "skeleton-dragon",
    "tornado",
    "zap",
]
spell_unit_list = [
    "arrows",
    "clone",
    "earthquake",
    "fireball",
    "freeze",
    "giant-snowball",
    "goblin-barrel",
    "graveyard",
    "lightning",
    "poison",
    "rage",
    "rocket",
    "skeleton-king-skill",
    "tesla-evolution-shock",
    "tornado",
    "zap",
    "the-log",
    "royal-delivery",
]
other_unit_list = [
    "bar",
    "bar-level",
    "clock",
    "dagger-duchess-tower-bar",
    "elixir",
    "emote",
    "evolution-symbol",
    "ice-spirit-evolution-symbol",
    "king-tower-bar",
    "skeleton-king-bar",
    "text",
    "tower-bar",
]
background_item_list = [
    "blood",
    "butterfly",
    "cup",
    "dagger-duchess-tower-icon",
    "flower",
    "grave",
    "crown-icon",
    "ribbon",
    "ruin",
    "king-tower-level",
    "king-tower-ruin",
    "skull",
    "scoreboard",
    "snow",
    "circle",
]
object_unit_list = [
    "axe",
    "dirt",
    "goblin-ball",
    "bomb",
]


def colorstr(s):
    return s


if __name__ == "__main__":
    check_union = set(ground_unit_list).intersection(flying_unit_list)
    assert (
        len(check_union) == 0
    ), f"Ground and fly should no intersection element {check_union}."
    avail_units = (
        set(ground_unit_list)
        | set(flying_unit_list)
        | set(other_unit_list)
        | set(tower_unit_list)
    )
    # avail_units.remove('bar-level')
    # unit_list.remove("selected")
    total_units = len(unit_list)
    print(colorstr(f"Total number unit n={total_units}"))
    # for i, u in enumerate(sorted(unit_list)):
    #   print(i+1, u, '✔' if u in avail_units else '✘')
    print(f"{colorstr(f'Available units (n={len(avail_units)})')}", sorted(avail_units))
    residue = set(unit_list) - avail_units
    print(f"{colorstr(f'Residue unit: (n={len(residue)})')}", sorted(residue))

    path_segment = Path("dataset/images/segment")
    segment_units = {"0": set(), "1": set()}
    for pu in path_segment.glob("*"):
        if pu.name in ["backgrounds", "background-items"]:
            continue
        for pi in pu.glob("*.png"):
            name = pi.stem
            sl = name.split("_")
            segment_units[sl[1]].add(sl[0])
    # print(segment_units['0'])
    # print(segment_units['1'])
    # print(segment_units['0'] - segment_units['1'])
    residue_set = set(unit_list) - {
        "text",
        "emote",
        "evolution-symbol",
        "dirt",
        "elixir",
        "axe",
        "dagger-duchess-tower-bar",
    }
    residue_set_1 = residue_set - segment_units["1"]
    print(
        f"{colorstr(f'Residue unit with 1 (n={len(residue_set_1)}, N-n={total_units-len(residue_set_1)})')}",
        sorted(residue_set_1),
    )
