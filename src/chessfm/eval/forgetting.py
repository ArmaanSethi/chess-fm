"""100-item deterministic general-English forgetting sanity benchmark.

Used to detect catastrophic forgetting or domain collapse (e.g., emitting chess
UCI moves or ``<think>`` traces on non-chess prompts) after Full Fine-Tuning
versus LoRA adaptation.

Contains exactly 100 items evenly split across 4 categories (25 each):
- ``reading_comprehension`` (MCQ A/B/C/D)
- ``logical_reasoning`` (MCQ A/B/C/D)
- ``world_knowledge`` (MCQ A/B/C/D)
- ``instruction_following`` (exact format / constraint checks)
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

CATEGORIES: tuple[str, ...] = (
    "reading_comprehension",
    "logical_reasoning",
    "world_knowledge",
    "instruction_following",
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*$", re.IGNORECASE | re.DOTALL)
_UCI_MOVE_RE = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b", re.IGNORECASE)
_CHESS_THINK_LEAK_RE = re.compile(
    r"(?:position assessment|best move|stockfish|centipawn|white to move|black to move)",
    re.IGNORECASE,
)
_CLOSE_THINK_RE = re.compile(r"</think>", re.IGNORECASE)
_MCQ_EXPLICIT_RE = re.compile(
    r"(?:^|\b)(?:the\s+)?(?:final\s+)?(?:answer|option|choice)\s*(?:is|:)?\s*"
    r"[*_`]*\(?\s*[*_`]*([A-D])[*_`]*\s*\)?[*_`]*\s*(?:[\.\)\:\,\!\?]|$|\n)",
    re.IGNORECASE,
)
_MCQ_STANDALONE_RE = re.compile(
    r"^\s*[*_`]*\(?\s*[*_`]*([A-D])[*_`]*\s*\)?[*_`]*[\.\)]?\s*$",
    re.IGNORECASE,
)
_MCQ_DELIMITED_LEADING_RE = re.compile(
    r"^\s*[*_`]*\(?[*_`]*([A-D])[*_`]*\)?[*_`]*[\.\)\:]\s",
    re.IGNORECASE,
)


def _mcq(
    item_id: str,
    category: str,
    question: str,
    choices: dict[str, str],
    answer: str,
) -> dict[str, Any]:
    lines = [question, ""]
    for key in ("A", "B", "C", "D"):
        lines.append(f"{key}. {choices[key]}")
    lines.append("")
    lines.append("Reply with only the single letter (A, B, C, or D).")
    return {
        "id": item_id,
        "category": category,
        "prompt": "\n".join(lines),
        "answer": answer.upper(),
        "answer_type": "mcq",
        "choices": dict(choices),
    }


def _exact(
    item_id: str,
    category: str,
    prompt: str,
    answer: str,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "category": category,
        "prompt": prompt,
        "answer": answer,
        "answer_type": "exact",
        "choices": {},
    }


def _build_reading_comprehension_items() -> list[dict[str, Any]]:
    raw: list[tuple[str, dict[str, str], str]] = [
        (
            "Passage: Mara planted three rows of tomatoes in April, but an unexpected late frost in early May destroyed the eastern row. She replaced those seedlings with basil because basil matures faster in warm soil.\nQuestion: Why did Mara choose basil to replace the damaged row?",
            {
                "A": "Basil resists frost better than tomatoes",
                "B": "Basil matures faster in warm soil",
                "C": "She ran out of tomato seeds in April",
                "D": "The eastern row received too little sunlight",
            },
            "B",
        ),
        (
            "Passage: The municipal library extended its weekday hours until 9:00 PM starting in October, while keeping Sunday hours unchanged at noon to 5:00 PM. Circulation of study guides rose by 18 percent the following month.\nQuestion: Which schedule remained the same in October?",
            {
                "A": "Weekday closing time",
                "B": "Saturday morning hours",
                "C": "Sunday operating hours",
                "D": "Holiday closures",
            },
            "C",
        ),
        (
            "Passage: Unlike copper, which corrodes into a green patina outdoors, stainless steel relies on a thin chromium-oxide film that reforms automatically when scratched in the presence of oxygen.\nQuestion: What enables stainless steel's protective layer to repair itself?",
            {
                "A": "Chromium reacting with oxygen",
                "B": "Copper alloying on the surface",
                "C": "Exposure to rainwater alone",
                "D": "Periodic waxing of the metal",
            },
            "A",
        ),
        (
            "Passage: During the 2024 harbour survey, biologists counted 42 osprey nests along the northern marshes, up from 29 five years earlier. They attributed the rebound primarily to the installation of elevated wooden nesting platforms.\nQuestion: What was the main cause cited for the increase in osprey nests?",
            {
                "A": "Warmer winter ocean currents",
                "B": "Bans on recreational boating",
                "C": "Elevated wooden nesting platforms",
                "D": "Higher fish stocks in the southern bay",
            },
            "C",
        ),
        (
            "Passage: Although the prototype drone could fly for 55 minutes without a payload, carrying the thermal camera reduced its flight endurance to 34 minutes. Engineers therefore scheduled battery swaps every 30 minutes to maintain a safety margin.\nQuestion: Why did engineers plan battery swaps every 30 minutes?",
            {
                "A": "The thermal camera overheats after 30 minutes",
                "B": "To keep a safety margin below the 34-minute loaded limit",
                "C": "Regulations forbid flights longer than half an hour",
                "D": "Charging stations are spaced 30 minutes apart",
            },
            "B",
        ),
        (
            "Passage: In 1859, a severe geomagnetic storm disrupted telegraph lines across Europe and North America. Some operators reported that induced currents were strong enough to send messages even after disconnecting their batteries.\nQuestion: What unusual phenomenon did telegraph operators observe during the 1859 storm?",
            {
                "A": "Telegraph wires melted into pools of iron",
                "B": "Messages could still be sent with batteries disconnected",
                "C": "Radio broadcasts drowned out Morse code",
                "D": "Submarine cables stopped working permanently",
            },
            "B",
        ),
        (
            "Passage: The museum's conservation team rejected harsh chemical solvents for cleaning the 17th-century tapestry and instead used controlled micro-vacuuming to remove soot without weakening the dyed wool fibers.\nQuestion: Why did conservators avoid chemical solvents?",
            {
                "A": "Solvents could weaken the dyed wool fibers",
                "B": "Solvents were too slow for a large tapestry",
                "C": "Micro-vacuuming restores faded colors automatically",
                "D": "Solvents only work on linen, not wool",
            },
            "A",
        ),
        (
            "Passage: Commuters on the River Line previously transferred twice to reach the airport. After the express shuttle opened in June, total travel time dropped from 78 minutes to 46 minutes with zero transfers.\nQuestion: How many minutes did the express shuttle save commuters?",
            {
                "A": "22 minutes",
                "B": "28 minutes",
                "C": "32 minutes",
                "D": "36 minutes",
            },
            "C",
        ),
        (
            "Passage: Desert beetles in Namibia harvest drinking water from morning coastal fog by tilting their wing covers forward so microscopic droplets coalesce and roll down toward their mouths.\nQuestion: How do the beetles collect water?",
            {
                "A": "By digging shallow wells beneath sand dunes",
                "B": "By condensing fog droplets on their wing covers",
                "C": "By storing rainwater inside hollow stems",
                "D": "By drinking directly from tidal pools",
            },
            "B",
        ),
        (
            "Passage: The software team migrated its logging pipeline from nightly batch jobs to a streaming queue. Alerts that once arrived up to 14 hours late now trigger within 90 seconds of an anomaly.\nQuestion: What improved as a result of switching to a streaming queue?",
            {
                "A": "Storage costs dropped to zero",
                "B": "Anomaly alerts trigger within 90 seconds instead of up to 14 hours",
                "C": "Engineers no longer need to write unit tests",
                "D": "Historical logs are automatically encrypted",
            },
            "B",
        ),
        (
            "Passage: When baking sourdough at high altitude, lower atmospheric pressure causes dough to rise more rapidly than at sea level. Bakers compensate by slightly reducing the starter percentage and shortening bulk fermentation.\nQuestion: How do high-altitude bakers adjust for faster rising?",
            {
                "A": "Add extra yeast and ferment longer",
                "B": "Lower oven temperature by 50 degrees",
                "C": "Reduce starter percentage and shorten bulk fermentation",
                "D": "Replace water entirely with milk",
            },
            "C",
        ),
        (
            "Passage: Before the bridge opened in 1962, ferries crossed the strait every forty minutes except during dense fog. Today, the ferry terminal serves as a maritime history center.\nQuestion: What is the former ferry terminal used for now?",
            {
                "A": "A commuter rail depot",
                "B": "A maritime history center",
                "C": "A coast guard radar station",
                "D": "A commercial fish market",
            },
            "B",
        ),
        (
            "Passage: Researchers found that urban street trees cooled shaded asphalt surfaces by up to 11 degrees Celsius on summer afternoons, though the cooling effect diminished sharply beyond 20 meters from the canopy.\nQuestion: What limitation did researchers note about the trees' cooling effect?",
            {
                "A": "It only occurs during winter mornings",
                "B": "It fades sharply more than 20 meters from the canopy",
                "C": "It increases humidity to hazardous levels",
                "D": "It works on concrete but not on asphalt",
            },
            "B",
        ),
        (
            "Passage: Elena audited 120 supplier invoices and discovered that 9 contained duplicate line items. All 9 duplicates originated from a single vendor whose billing system had been upgraded the week prior.\nQuestion: What did all nine erroneous invoices have in common?",
            {
                "A": "They were submitted more than six months late",
                "B": "They lacked purchase order numbers",
                "C": "They came from one vendor after a billing system upgrade",
                "D": "They were billed in foreign currency",
            },
            "C",
        ),
        (
            "Passage: Coral polyps obtain most of their energy from microscopic algae living inside their tissues. When ocean temperatures stay abnormally high, corals expel these algae, turning white in a process called bleaching.\nQuestion: What directly causes bleached coral to turn white?",
            {
                "A": "Accumulation of sea salt on the reef",
                "B": "Expulsion of the symbiotic algae from coral tissues",
                "C": "Overgrowth of kelp blocking sunlight",
                "D": "Rapid calcification during cold snaps",
            },
            "B",
        ),
        (
            "Passage: The orchestra rehearsed in the smaller recital hall while the main auditorium underwent acoustic paneling repairs. Violists noted that the smaller room made ensemble balance easier to hear.\nQuestion: Why was the orchestra rehearsing in the recital hall?",
            {
                "A": "Ticket sales were too low for the main hall",
                "B": "The main auditorium was having acoustic paneling repaired",
                "C": "The recital hall has a larger stage",
                "D": "Air conditioning failed in the main auditorium",
            },
            "B",
        ),
        (
            "Passage: Paper maps remain mandatory backup equipment on commercial vessels because electronic chart displays depend on electrical power and satellite signals that can fail simultaneously during severe storms.\nQuestion: Why are paper charts still required on ships?",
            {
                "A": "Electronic charts are less accurate in calm weather",
                "B": "Paper maps do not rely on power or satellite signals",
                "C": "Port authorities refuse digital logbooks",
                "D": "Satellite navigation does not cover coastal waters",
            },
            "B",
        ),
        (
            "Passage: In a controlled trial, students who spaced their vocabulary review across four short sessions recalled 74 percent of words a month later, compared with 48 percent for students who studied the same total duration in one sitting.\nQuestion: Which study strategy yielded higher one-month recall?",
            {
                "A": "Studying all words in a single intensive session",
                "B": "Spacing review across four short sessions",
                "C": "Reading definitions aloud without testing",
                "D": "Doubling the total number of study hours",
            },
            "B",
        ),
        (
            "Passage: Geothermal heat pumps circulate fluid through buried ground loops where soil temperatures stay nearly constant year-round, allowing them to heat buildings in winter and cool them in summer efficiently.\nQuestion: Why are buried ground loops effective for heat pumps?",
            {
                "A": "Underground soil maintains a nearly constant temperature all year",
                "B": "Groundwater boils at shallow depths",
                "C": "Buried pipes require no electricity to circulate fluid",
                "D": "Soil prevents metal pipes from rusting",
            },
            "A",
        ),
        (
            "Passage: The archive digitized 1,400 glass-plate negatives from the 1890s. Because the emulsion flakes easily when handled, technicians photographed each plate on a lightbox rather than feeding it through a roller scanner.\nQuestion: Why did technicians avoid using a roller scanner?",
            {
                "A": "Roller scanners cannot capture monochrome images",
                "B": "Handling could cause the fragile emulsion to flake off",
                "C": "Lightboxes are faster than automated scanners",
                "D": "Glass plates are too thin for roller feeds",
            },
            "B",
        ),
        (
            "Passage: Every Thursday, the bakery donates unsold loaves to two neighborhood shelters. Since introducing smaller afternoon baking batches in March, unsold inventory fell by 40 percent without reducing customer sales.\nQuestion: What effect did smaller afternoon batches have?",
            {
                "A": "Customer sales declined noticeably",
                "B": "Unsold loaves dropped by 40 percent while sales held steady",
                "C": "The bakery stopped opening on Thursdays",
                "D": "Bread prices had to be raised by 40 percent",
            },
            "B",
        ),
        (
            "Passage: Seismographs on the volcanic island detected a swarm of shallow tremors starting Tuesday morning. Although no eruption occurred, geologists raised the alert level from green to yellow as a precaution.\nQuestion: What prompted geologists to raise the alert level to yellow?",
            {
                "A": "Ash plumes reached commercial flight altitudes",
                "B": "Lava reached the coastal highway",
                "C": "A swarm of shallow tremors began Tuesday morning",
                "D": "Satellite imagery showed a collapsed crater",
            },
            "C",
        ),
        (
            "Passage: Unlike deciduous oaks that shed leaves each autumn, coastal live oaks retain green foliage throughout winter and replace leaves gradually in spring.\nQuestion: When do coastal live oaks replace their leaves?",
            {
                "A": "All at once in late autumn",
                "B": "Gradually in spring",
                "C": "Only during drought years",
                "D": "Never; the same leaves last decades",
            },
            "B",
        ),
        (
            "Passage: The city replaced 3,200 sodium-vapor streetlights with shielded LED fixtures. Nighttime electricity consumption fell by half, and astronomers at the nearby observatory reported noticeably darker skies.\nQuestion: Which two benefits followed the streetlight replacement?",
            {
                "A": "Halved electricity use and darker skies for the observatory",
                "B": "Brighter glare and warmer yellow street illumination",
                "C": "Zero maintenance costs and faster traffic speeds",
                "D": "Higher daytime solar generation and quieter roads",
            },
            "A",
        ),
        (
            "Passage: When honeybees discover a rich patch of clover more than 100 meters from the hive, they perform a waggle dance whose angle relative to vertical encodes the direction of the food relative to the sun.\nQuestion: What does the angle of the waggle dance communicate?",
            {
                "A": "The sugar concentration of the nectar",
                "B": "The direction of the food source relative to the sun",
                "C": "The number of predators near the flowers",
                "D": "The altitude of the clover field",
            },
            "B",
        ),
    ]
    return [
        _mcq(f"rc_{i:02d}", "reading_comprehension", q, ch, ans)
        for i, (q, ch, ans) in enumerate(raw, start=1)
    ]


def _build_logical_reasoning_items() -> list[dict[str, Any]]:
    raw: list[tuple[str, dict[str, str], str]] = [
        (
            "All cedar trees are conifers. All conifers produce cones. Therefore, which statement MUST be true?",
            {
                "A": "All cone-producing plants are cedar trees",
                "B": "All cedar trees produce cones",
                "C": "Some cedar trees do not produce cones",
                "D": "No pine trees are conifers",
            },
            "B",
        ),
        (
            "If it rains tonight, the trail will be muddy tomorrow. The trail is not muddy tomorrow. What can we logically conclude?",
            {
                "A": "It rained lightly tonight",
                "B": "It did not rain tonight",
                "C": "The trail was paved overnight",
                "D": "Nothing can be concluded",
            },
            "B",
        ),
        (
            "In a race among four runners, Alice finished ahead of Ben, Ben finished ahead of Cara, and Dan finished ahead of Alice. Who finished second?",
            {
                "A": "Alice",
                "B": "Ben",
                "C": "Cara",
                "D": "Dan",
            },
            "A",
        ),
        (
            "What is the next number in the sequence: 3, 6, 12, 24, 48, ...?",
            {
                "A": "72",
                "B": "84",
                "C": "96",
                "D": "108",
            },
            "C",
        ),
        (
            "No reptiles are warm-blooded. All iguanas are reptiles. Which conclusion follows necessarily?",
            {
                "A": "Some iguanas are warm-blooded",
                "B": "No iguanas are warm-blooded",
                "C": "All cold-blooded animals are iguanas",
                "D": "Some reptiles are mammals",
            },
            "B",
        ),
        (
            "A store offers a 20% discount on a $50 jacket. What is the sale price?",
            {
                "A": "$30",
                "B": "$35",
                "C": "$40",
                "D": "$45",
            },
            "C",
        ),
        (
            "Three boxes are labeled Apples, Oranges, and Mixed, and EVERY label is wrong. You draw one fruit from the box labeled Mixed and it is an apple. What is actually inside the box labeled Oranges?",
            {
                "A": "Apples",
                "B": "Oranges",
                "C": "Mixed",
                "D": "Cannot be determined",
            },
            "C",
        ),
        (
            "If every Tuesday meeting is canceled whenever the director travels, and the director is traveling this Tuesday, what happens to this Tuesday's meeting?",
            {
                "A": "It is moved to Wednesday",
                "B": "It is canceled",
                "C": "It is led by the assistant",
                "D": "It is shortened by half",
            },
            "B",
        ),
        (
            "A train leaves at 14:20 and arrives at 17:05 on the same day. How long is the journey?",
            {
                "A": "2 hours 35 minutes",
                "B": "2 hours 45 minutes",
                "C": "3 hours 15 minutes",
                "D": "3 hours 25 minutes",
            },
            "B",
        ),
        (
            "Which number does NOT belong in the sequence of squares: 4, 9, 16, 24, 36?",
            {
                "A": "9",
                "B": "16",
                "C": "24",
                "D": "36",
            },
            "C",
        ),
        (
            "Suppose 'If P then Q' is true and 'If Q then R' is true. If P is true, which of the following MUST be true?",
            {
                "A": "R is true",
                "B": "R is false",
                "C": "Q is false",
                "D": "Not-P is true",
            },
            "A",
        ),
        (
            "A rectangle has perimeter 30 cm and width 6 cm. What is its length?",
            {
                "A": "8 cm",
                "B": "9 cm",
                "C": "12 cm",
                "D": "18 cm",
            },
            "B",
        ),
        (
            "Five machines make 5 widgets in 5 minutes. At the same constant rate, how long does it take 100 machines to make 100 widgets?",
            {
                "A": "1 minute",
                "B": "5 minutes",
                "C": "20 minutes",
                "D": "100 minutes",
            },
            "B",
        ),
        (
            "If today is Wednesday, what day of the week will it be 10 days from now?",
            {
                "A": "Friday",
                "B": "Saturday",
                "C": "Sunday",
                "D": "Monday",
            },
            "B",
        ),
        (
            "All librarians at Oakwood speak French. Some Oakwood staff are librarians. Which statement MUST be true?",
            {
                "A": "All Oakwood staff speak French",
                "B": "Some Oakwood staff speak French",
                "C": "No Oakwood staff speak Spanish",
                "D": "Everyone who speaks French is a librarian",
            },
            "B",
        ),
        (
            "A jar contains 4 red marbles, 5 blue marbles, and 3 green marbles. If you draw one marble at random, what is the probability it is green?",
            {
                "A": "1/6",
                "B": "1/4",
                "C": "1/3",
                "D": "5/12",
            },
            "B",
        ),
        (
            "What is the next number in the Fibonacci-like sequence: 2, 3, 5, 8, 13, 21, ...?",
            {
                "A": "29",
                "B": "32",
                "C": "34",
                "D": "36",
            },
            "C",
        ),
        (
            "Maya is taller than Noah. Owen is shorter than Noah. Priya is taller than Maya. Who is the shortest?",
            {
                "A": "Maya",
                "B": "Noah",
                "C": "Owen",
                "D": "Priya",
            },
            "C",
        ),
        (
            "A tank fills at 6 liters per minute and leaks at 2 liters per minute. How many minutes does it take to fill an empty 40-liter tank?",
            {
                "A": "5 minutes",
                "B": "8 minutes",
                "C": "10 minutes",
                "D": "20 minutes",
            },
            "C",
        ),
        (
            "Which statement is the logical contrapositive of 'If a number is divisible by 4, then it is even'?",
            {
                "A": "If a number is even, then it is divisible by 4",
                "B": "If a number is not even, then it is not divisible by 4",
                "C": "If a number is not divisible by 4, then it is not even",
                "D": "Some even numbers are divisible by 4",
            },
            "B",
        ),
        (
            "A bookshelf holds 7 books in a row. How many books are between the 2nd book from the left and the 6th book from the left?",
            {
                "A": "2",
                "B": "3",
                "C": "4",
                "D": "5",
            },
            "B",
        ),
        (
            "If 3x + 7 = 22, what is the value of 2x - 1?",
            {
                "A": "7",
                "B": "9",
                "C": "11",
                "D": "13",
            },
            "B",
        ),
        (
            "Two fair six-sided dice are rolled. Which sum is MOST likely?",
            {
                "A": "2",
                "B": "5",
                "C": "7",
                "D": "11",
            },
            "C",
        ),
        (
            "Every member of Club X is also in Club Y. Sam is NOT in Club Y. What follows?",
            {
                "A": "Sam is in Club X",
                "B": "Sam is not in Club X",
                "C": "Sam might be in Club X",
                "D": "Club Y is empty",
            },
            "B",
        ),
        (
            "A cyclist rides 18 km at 12 km/h. How many minutes does the ride take?",
            {
                "A": "60 minutes",
                "B": "75 minutes",
                "C": "90 minutes",
                "D": "105 minutes",
            },
            "C",
        ),
    ]
    return [
        _mcq(f"lr_{i:02d}", "logical_reasoning", q, ch, ans)
        for i, (q, ch, ans) in enumerate(raw, start=1)
    ]


def _build_world_knowledge_items() -> list[dict[str, Any]]:
    raw: list[tuple[str, dict[str, str], str]] = [
        (
            "What is the chemical symbol for gold?",
            {"A": "Ag", "B": "Au", "C": "Gd", "D": "Go"},
            "B",
        ),
        (
            "Which planet in our solar system is known for its prominent ring system and is the sixth planet from the Sun?",
            {"A": "Jupiter", "B": "Saturn", "C": "Uranus", "D": "Neptune"},
            "B",
        ),
        (
            "What is the capital city of Japan?",
            {"A": "Osaka", "B": "Kyoto", "C": "Tokyo", "D": "Nagoya"},
            "C",
        ),
        (
            "Which gas makes up approximately 78 percent of Earth's atmosphere?",
            {"A": "Oxygen", "B": "Carbon dioxide", "C": "Nitrogen", "D": "Argon"},
            "C",
        ),
        (
            "Who wrote the play 'Hamlet'?",
            {
                "A": "Christopher Marlowe",
                "B": "William Shakespeare",
                "C": "Ben Jonson",
                "D": "John Milton",
            },
            "B",
        ),
        (
            "Which organelle is primarily responsible for generating ATP in eukaryotic cells?",
            {"A": "Ribosome", "B": "Golgi apparatus", "C": "Mitochondrion", "D": "Lysosome"},
            "C",
        ),
        (
            "Which river flows through Cairo, Egypt?",
            {"A": "Tigris", "B": "Congo", "C": "Nile", "D": "Zambezi"},
            "C",
        ),
        (
            "At standard sea-level atmospheric pressure, at what temperature does pure water boil on the Celsius scale?",
            {"A": "90 °C", "B": "100 °C", "C": "112 °C", "D": "212 °C"},
            "B",
        ),
        (
            "Which continent contains the Sahara Desert?",
            {"A": "Asia", "B": "Australia", "C": "Africa", "D": "South America"},
            "C",
        ),
        (
            "What is the SI unit of electrical resistance?",
            {"A": "Volt", "B": "Ampere", "C": "Ohm", "D": "Watt"},
            "C",
        ),
        (
            "Which scientist formulated the three laws of motion published in the 'Principia' (1687)?",
            {
                "A": "Galileo Galilei",
                "B": "Isaac Newton",
                "C": "Johannes Kepler",
                "D": "Michael Faraday",
            },
            "B",
        ),
        (
            "Which ocean is the largest by surface area?",
            {"A": "Atlantic Ocean", "B": "Indian Ocean", "C": "Pacific Ocean", "D": "Arctic Ocean"},
            "C",
        ),
        (
            "How many chromosomes are normally found in a human somatic (non-reproductive) cell?",
            {"A": "23", "B": "44", "C": "46", "D": "48"},
            "C",
        ),
        (
            "Which element has atomic number 6 and forms the backbone of organic molecules?",
            {"A": "Silicon", "B": "Carbon", "C": "Oxygen", "D": "Nitrogen"},
            "B",
        ),
        (
            "In what year did humans first land on the Moon during the Apollo 11 mission?",
            {"A": "1965", "B": "1967", "C": "1969", "D": "1972"},
            "C",
        ),
        (
            "Which mountain range separates Europe from Asia across Russia?",
            {"A": "Alps", "B": "Andes", "C": "Ural Mountains", "D": "Appalachian Mountains"},
            "C",
        ),
        (
            "What type of blood vessel carries blood AWAY from the heart?",
            {"A": "Vein", "B": "Artery", "C": "Venule", "D": "Lymph duct"},
            "B",
        ),
        (
            "Which classical composer wrote the 'Moonlight Sonata' and Symphony No. 9?",
            {
                "A": "Wolfgang Amadeus Mozart",
                "B": "Johann Sebastian Bach",
                "C": "Ludwig van Beethoven",
                "D": "Franz Schubert",
            },
            "C",
        ),
        (
            "What is the hardest naturally occurring mineral on the Mohs hardness scale?",
            {"A": "Quartz", "B": "Topaz", "C": "Corundum", "D": "Diamond"},
            "D",
        ),
        (
            "Which country is the city of Reykjavik the capital of?",
            {"A": "Norway", "B": "Iceland", "C": "Finland", "D": "Denmark"},
            "B",
        ),
        (
            "What process do green plants use to convert sunlight, water, and carbon dioxide into glucose and oxygen?",
            {"A": "Fermentation", "B": "Photosynthesis", "C": "Osmosis", "D": "Transpiration"},
            "B",
        ),
        (
            "Which layer of Earth lies directly beneath the crust?",
            {"A": "Inner core", "B": "Outer core", "C": "Mantle", "D": "Troposphere"},
            "C",
        ),
        (
            "Who painted the 'Mona Lisa'?",
            {"A": "Michelangelo", "B": "Raphael", "C": "Leonardo da Vinci", "D": "Botticelli"},
            "C",
        ),
        (
            "Which programming data structure operates on a Last-In, First-Out (LIFO) principle?",
            {"A": "Queue", "B": "Stack", "C": "Hash table", "D": "Linked list"},
            "B",
        ),
        (
            "Which strait connects the Mediterranean Sea to the Atlantic Ocean?",
            {
                "A": "Strait of Hormuz",
                "B": "Bering Strait",
                "C": "Strait of Gibraltar",
                "D": "Strait of Malacca",
            },
            "C",
        ),
    ]
    return [
        _mcq(f"wk_{i:02d}", "world_knowledge", q, ch, ans)
        for i, (q, ch, ans) in enumerate(raw, start=1)
    ]


def _build_instruction_following_items() -> list[dict[str, Any]]:
    raw: list[tuple[str, str]] = [
        ("Reply with the exact word BLUE in all uppercase and nothing else.", "BLUE"),
        ("Reverse the letters of the word 'garden' and output only the reversed word.", "nedrag"),
        (
            "List the first four prime numbers separated by commas with no spaces (e.g. a,b,c,d).",
            "2,3,5,7",
        ),
        (
            "Convert the phrase 'hello world' to uppercase and output only the result.",
            "HELLO WORLD",
        ),
        ("Output the exact integer result of 17 * 6 and nothing else.", "102"),
        (
            "Sort these three words alphabetically and join them with hyphens: pear, apple, mango.",
            "apple-mango-pear",
        ),
        ("Reply with ONLY the word 'yes' in lowercase if 15 is greater than 9.", "yes"),
        (
            "Count the number of letter 'a' characters in 'banana' and output only the integer.",
            "3",
        ),
        ("Output the third month of the year in lowercase only.", "march"),
        (
            "Remove all vowels (a, e, i, o, u) from the word 'planet' and output only the result.",
            "plnt",
        ),
        (
            "Format the date January 5, 2026 as YYYY-MM-DD and output nothing else.",
            "2026-01-05",
        ),
        (
            "Output the acronym formed by the first letters of 'North Atlantic Treaty Organization'.",
            "NATO",
        ),
        ("Compute 144 divided by 12 and reply with only the integer.", "12"),
        ("Write the word 'echo' twice separated by a single space and nothing else.", "echo echo"),
        (
            "Return the last word of the sentence 'The silver train arrived early' in lowercase.",
            "early",
        ),
        ("Output the Roman numeral for 9 in uppercase and nothing else.", "IX"),
        (
            "Replace every space in 'red fox jumps' with an underscore and output only the result.",
            "red_fox_jumps",
        ),
        ("Output the square of 13 as a plain integer with no punctuation.", "169"),
        (
            "Write the first five letters of the English alphabet in lowercase with no spaces.",
            "abcde",
        ),
        ("Output the opposite of 'north' in lowercase and nothing else.", "south"),
        (
            "Extract only the digits from 'ID-482-X9' in order and output them as one string.",
            "4829",
        ),
        ("Output the JSON array containing the integers 1, 2, and 3 with no spaces.", "[1,2,3]"),
        ("Capitalize only the first letter of 'python' and output nothing else.", "Python"),
        ("Output the sum of 25, 35, and 40 as a single integer.", "100"),
        ("Reply with the exact string OK-200 and nothing else.", "OK-200"),
    ]
    return [
        _exact(f"if_{i:02d}", "instruction_following", prompt, ans)
        for i, (prompt, ans) in enumerate(raw, start=1)
    ]


def _rebalance_mcq_items(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Permute choice slots deterministically so A/B/C/D keys are evenly balanced."""
    keys = ("A", "B", "C", "D")
    balanced: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        target_key = keys[idx % 4]
        old_key = item["answer"]
        choices = dict(item["choices"])
        if old_key != target_key:
            choices[old_key], choices[target_key] = choices[target_key], choices[old_key]
        question = item["prompt"].split("\n\nA. ", 1)[0]
        balanced.append(_mcq(item["id"], item["category"], question, choices, target_key))
    return balanced


_BENCHMARK_ITEMS: tuple[dict[str, Any], ...] = tuple(
    _rebalance_mcq_items(
        _build_reading_comprehension_items()
        + _build_logical_reasoning_items()
        + _build_world_knowledge_items()
    )
    + _build_instruction_following_items()
)


def get_forgetting_benchmark() -> list[dict[str, Any]]:
    """Return deep copies of all 100 general-English forgetting benchmark items."""
    return [copy.deepcopy(item) for item in _BENCHMARK_ITEMS]


def export_forgetting_jsonl(output_path: str | Path) -> int:
    """Write all 100 benchmark items to ``output_path`` as JSONL and return count."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = get_forgetting_benchmark()
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(items)


def _strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` blocks and orphan closing ``</think>`` prefixes."""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    if _CLOSE_THINK_RE.search(cleaned):
        cleaned = _CLOSE_THINK_RE.split(cleaned)[-1]
    cleaned = _UNCLOSED_THINK_RE.sub("", cleaned)
    return cleaned.strip()


def _detect_chess_leakage(raw_response: str, stripped_response: str, expected: str) -> bool:
    """Return True if a non-chess answer collapses into chess move / trace output."""
    if not raw_response:
        return False

    raw_lower = raw_response.lower()
    has_think_tag = "<think>" in raw_lower or "</think>" in raw_lower
    uci_matches = _UCI_MOVE_RE.findall(stripped_response)
    has_chess_jargon = bool(_CHESS_THINK_LEAK_RE.search(raw_response))

    # Pure UCI move output (e.g. "e2e4" or "<think>...</think>\ne2e4") on a general-English prompt.
    if stripped_response and _UCI_MOVE_RE.fullmatch(stripped_response):
        if stripped_response.lower() != expected.strip().lower():
            return True

    if has_think_tag and (uci_matches or has_chess_jargon):
        return True

    if len(uci_matches) >= 2 and has_chess_jargon:
        return True

    return False


def _extract_mcq_letter(text: str) -> str:
    """Extract a normalized A/B/C/D choice from a model response."""
    if not text:
        return ""
    standalone = _MCQ_STANDALONE_RE.match(text)
    if standalone:
        return standalone.group(1).upper()
    explicit = _MCQ_EXPLICIT_RE.search(text)
    if explicit:
        return explicit.group(1).upper()
    leading = _MCQ_DELIMITED_LEADING_RE.match(text)
    if leading:
        return leading.group(1).upper()
    return ""


def _normalize_exact_answer(text: str) -> str:
    """Normalize whitespace, markdown wrappers, and surrounding quotes for exact checks."""
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    for wrapper in ("**", "__"):
        if (
            len(first_line) > 2 * len(wrapper)
            and first_line.startswith(wrapper)
            and first_line.endswith(wrapper)
        ):
            first_line = first_line[len(wrapper) : -len(wrapper)].strip()
    if (
        len(first_line) >= 2
        and first_line[0] == first_line[-1]
        and first_line[0] in ("'", '"', "`")
    ):
        first_line = first_line[1:-1].strip()
    return first_line


def score_forgetting_response(item: dict[str, Any], raw_response: str) -> dict[str, Any]:
    """Score a single response against a forgetting-benchmark item."""
    raw = "" if raw_response is None else str(raw_response)
    stripped = _strip_think_tags(raw)
    expected = str(item["answer"])
    answer_type = str(item.get("answer_type", "exact"))

    if answer_type == "mcq":
        predicted = _extract_mcq_letter(stripped)
        correct = predicted == expected.strip().upper()
    else:
        predicted = _normalize_exact_answer(stripped)
        correct = predicted == expected.strip()

    chess_leakage = _detect_chess_leakage(raw, stripped, expected)

    return {
        "id": item["id"],
        "category": item["category"],
        "answer_type": answer_type,
        "expected": expected,
        "predicted": predicted,
        "correct": bool(correct),
        "chess_leakage": bool(chess_leakage),
    }


def evaluate_forgetting_benchmark(
    answer_fn: Callable[[dict[str, Any]], str],
    items: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run ``answer_fn`` on the forgetting benchmark and return summary metrics."""
    bench_items = list(items) if items is not None else get_forgetting_benchmark()
    results: list[dict[str, Any]] = []
    per_cat_counts: dict[str, dict[str, int]] = {}

    for item in bench_items:
        raw = answer_fn(item)
        scored = score_forgetting_response(item, raw)
        results.append(scored)

        cat = scored["category"]
        bucket = per_cat_counts.setdefault(cat, {"total": 0, "correct": 0, "chess_leakage": 0})
        bucket["total"] += 1
        if scored["correct"]:
            bucket["correct"] += 1
        if scored["chess_leakage"]:
            bucket["chess_leakage"] += 1

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    leaked = sum(1 for r in results if r["chess_leakage"])

    per_category: dict[str, dict[str, Any]] = {}
    for cat, counts in per_cat_counts.items():
        cat_total = counts["total"]
        per_category[cat] = {
            "total": cat_total,
            "correct": counts["correct"],
            "accuracy": counts["correct"] / cat_total if cat_total else 0.0,
            "chess_leakage_rate": counts["chess_leakage"] / cat_total if cat_total else 0.0,
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "chess_leakage_rate": leaked / total if total else 0.0,
        "per_category": per_category,
        "results": results,
    }
