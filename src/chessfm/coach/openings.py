"""ECO opening book, transposition-aware FEN index, and opening plan lookup.

Provides a deterministic, zero-network opening repertoire covering 42 classic
1.e4, 1.d4, and flank openings and tactical traps. Every entry's canonical
4-field board FEN is derived by replaying its SAN move sequence from
``chess.STARTING_FEN`` via ``python-chess``, guaranteeing 100% legal positions
and automatic transposition matching.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import chess

from chessfm.data_generation.phase_sampler import normalize_board_fen


@dataclass(frozen=True)
class OpeningEntry:
    """A single opening line with strategic plans and a signature tactical trap."""

    eco: str
    name: str
    moves: tuple[str, ...]
    white_plan: str
    black_plan: str
    common_trap: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entry to a JSON-friendly dictionary."""
        return {
            "eco": self.eco,
            "name": self.name,
            "moves": list(self.moves),
            "white_plan": self.white_plan,
            "black_plan": self.black_plan,
            "common_trap": self.common_trap,
        }


OPENING_BOOK: tuple[OpeningEntry, ...] = (
    # ------------------------------------------------------------------
    # 1.e4 e5 — Open Games
    # ------------------------------------------------------------------
    OpeningEntry(
        eco="C50",
        name="Italian Game",
        moves=("e4", "e5", "Nf3", "Nc6", "Bc4"),
        white_plan="Control the center with c3 and d4 while targeting f7 with the light-squared bishop.",
        black_plan="Develop harmoniously with Bc5 or Nf6 and contest the d4 break.",
        common_trap="Playing 3...Nd4? invites the Blackburne Shilling Gambit (4. Nxe5 Qg5!), where 5. Nxf7 Qxg2 wins material if White gets greedy.",
    ),
    OpeningEntry(
        eco="C53",
        name="Giuoco Piano",
        moves=("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"),
        white_plan="Build a classical center via c3 and d4 or play the slow Giuoco Pianissimo with d3, c3, and Nbd2-f1-g3.",
        black_plan="Castle early, keep the e5 stronghold intact, and counter in the center with ...d5 when White's pieces are uncoordinated.",
        common_trap="After 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 7. Nc3 Nxe4, both sides must calculate carefully or lose a piece on c3/e4.",
    ),
    OpeningEntry(
        eco="C55",
        name="Two Knights Defense",
        moves=("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6"),
        white_plan="Choose between the sharp 4. Ng5 targeting f7 or the solid 4. d3 / 4. d4 central push.",
        black_plan="Counterattack the e4 pawn immediately and seize active piece play if White grabs pawns.",
        common_trap="After 4. Ng5 d5 5. exd5, grabbing back with 5...Nxd5? walks into the Fried Liver Attack instead of the main line 5...Na5!",
    ),
    OpeningEntry(
        eco="C57",
        name="Fried Liver Attack",
        moves=("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Ng5", "d5", "exd5", "Nxd5", "Nxf7"),
        white_plan="Follow 6...Kxf7 with 7. Qf3+ Ke6 8. Nc3 to pin the d5 knight and hunt the exposed Black king.",
        black_plan="Defend the d5 knight with ...Ke6 and ...Nb4/c6 while looking to consolidate the extra piece.",
        common_trap="Retreating 7...Kg8? or 7...Ke8? after 7. Qf3+ immediately drops the d5 knight to Bxd5+ with a crushing attack.",
    ),
    OpeningEntry(
        eco="C60",
        name="Ruy Lopez",
        moves=("e4", "e5", "Nf3", "Nc6", "Bb5"),
        white_plan="Exert long-term pressure on e5, castle quickly, and prepare c3 and d4 for a strong central majority.",
        black_plan="Challenge the bishop with ...a6 or counterattack e4 with ...Nf6 (Berlin) while maintaining solid central control.",
        common_trap="In the Noah's Ark Trap, White's light-squared bishop gets trapped on b3 after ...a6, ...b5, ...d6, and ...c5-c4 if White grabs d4 carelessly.",
    ),
    OpeningEntry(
        eco="C78",
        name="Ruy Lopez: Morphy Defense",
        moves=("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O"),
        white_plan="Play Re1, c3, and d4 (or a4 in the Anti-Marshall) to build a lasting kingside and central initiative.",
        black_plan="Develop with ...Be7, ...b5, and ...O-O, aiming for counterplay via the Marshall Attack (...d5) or Chigorin setup.",
        common_trap="In the Open Ruy Lopez (5...Nxe4 6. d4 b5 7. Bb3 d5 8. dxe5 Be6), grabbing c6 too early or ignoring ...Nc5 leaves White's bishop exposed.",
    ),
    OpeningEntry(
        eco="C45",
        name="Scotch Game",
        moves=("e4", "e5", "Nf3", "Nc6", "d4", "exd4", "Nxd4"),
        white_plan="Open the center immediately for active minor pieces and a space advantage on the kingside.",
        black_plan="Develop dynamically with 4...Bc5 or 4...Nf6 5. Nxc6 bxc6 to fight for rapid piece activity.",
        common_trap="After 4...Bc5 5. Be3 Qf6 6. c3 Nge7, careless moves like 5. Nxc6? Qf6! threaten mate on f2 and punish White's premature trade.",
    ),
    OpeningEntry(
        eco="C42",
        name="Petrov's Defense",
        moves=("e4", "e5", "Nf3", "Nf6"),
        white_plan="Capture 3. Nxe5 and fight for a slight structural edge after 3...d6 4. Nf3 Nxe4 5. d4 d5 6. Bd3.",
        black_plan="Neutralize White's opening initiative with symmetrical piece activity and a rock-solid central structure.",
        common_trap="Copying White with 3. Nxe5 Nxe4? 4. Qe2! loses the e4 knight or the queen if Black retreats 4...Nf6? allowing 5. Nc6+!",
    ),
    OpeningEntry(
        eco="C42",
        name="Stafford Gambit",
        moves=("e4", "e5", "Nf3", "Nf6", "Nxe5", "Nc6", "Nxc6", "dxc6"),
        white_plan="Defend e4 solidly with d3 or Nc3, avoid weakening f2/g2, and convert the extra pawn in a calm position.",
        black_plan="Launch an all-out kingside attack with ...Bc5, ...Ng4, ...Qh4, and ...h5 before White consolidates.",
        common_trap="After 5. d3 Bc5 6. Bg5? Nxe4! 7. Bxd8?? Bxf2+ 8. Ke2 Bg4# is a famous smothered-style mating net.",
    ),
    OpeningEntry(
        eco="C41",
        name="Philidor Defense",
        moves=("e4", "e5", "Nf3", "d6"),
        white_plan="Strike the center with 3. d4 and use the space advantage to coordinate pieces against Black's cramped position.",
        black_plan="Hold the e5 point firmly with ...Nd7 and ...Ngf6, aiming for a resilient counterattacking setup.",
        common_trap="Légal's Mate arises if Black pins the knight with ...Bg4 and grabs the queen on d1 after Nxe5, allowing Bxf7+ Ke7 Nd5#.",
    ),
    OpeningEntry(
        eco="C30",
        name="King's Gambit",
        moves=("e4", "e5", "f4"),
        white_plan="Deflect Black's e5 pawn to build a massive center with d4 and open the f-file for a kingside assault.",
        black_plan="Accept with 2...exf4 and counter in the center with ...d5, or decline cleanly with 2...Bc5 or 2...d5 (Falkbeer).",
        common_trap="After 2...exf4 3. Nf3 g5 4. Bc4 g4, White's Muzio Gambit (5. O-O gxf3 6. Qxf3) punishes greedy defense if Black neglects development.",
    ),
    OpeningEntry(
        eco="C25",
        name="Vienna Game",
        moves=("e4", "e5", "Nc3"),
        white_plan="Prepare a well-supported f4 break (Vienna Gambit) or develop with Bc4 and d3 for flexible kingside pressure.",
        black_plan="Respond actively with 2...Nf6 and meet 3. f4 with the principled central counter 3...d5!",
        common_trap="In the Vienna Gambit (2...Nf6 3. f4), accepting with 3...exf4? runs into 4. e5!, trapping or displacing Black's knight awkwardly.",
    ),
    OpeningEntry(
        eco="C20",
        name="Scholar's Mate Trap",
        moves=("e4", "e5", "Bc4", "Nc6", "Qh5"),
        white_plan="Threaten immediate checkmate on f7 while pressuring e5; if parried, retreat the queen safely and develop.",
        black_plan="Defend mate cleanly with 3...g6 4. Qf3 Nf6, gaining tempo against White's early queen sortie.",
        common_trap="Playing 3...Nf6?? to attack the queen overlooks 4. Qxf7# checkmate on the spot.",
    ),
    # ------------------------------------------------------------------
    # 1.e4 — Semi-Open Games (Sicilian, French, Caro-Kann, etc.)
    # ------------------------------------------------------------------
    OpeningEntry(
        eco="B20",
        name="Sicilian Defense",
        moves=("e4", "c5"),
        white_plan="Choose between the Open Sicilian (Nf3 and d4) for dynamic attacking play or Anti-Sicilians like c3, Nc3, or Bb5.",
        black_plan="Create an asymmetrical fight for the d4 square and generate queenside counterplay along the half-open c-file.",
        common_trap="Overextending prematurely with ...d5 without support allows White to open lines against an uncastled Black king.",
    ),
    OpeningEntry(
        eco="B50",
        name="Open Sicilian",
        moves=("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4"),
        white_plan="Use the central space and rapid piece development to mount a kingside attack before Black's queenside play arrives.",
        black_plan="Develop with ...Nf6, contest the center with ...e5 or ...e6, and exploit the c-file and extra central pawn.",
        common_trap="Misplacing pieces on the a4-e8 diagonal allows tactical checks like Bb5+ winning material or disrupting castling.",
    ),
    OpeningEntry(
        eco="B90",
        name="Sicilian Defense: Najdorf Variation",
        moves=("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"),
        white_plan="Attack aggressively with Be3/f3 (English Attack), Bg5, or Be2 while targeting the d5 outpost.",
        black_plan="Control b5, play ...e5 or ...e6, expand on the queenside with ...b5, and counterattack White's e4 pawn.",
        common_trap="In the Poisoned Pawn line (6. Bg5 e6 7. f4 Qb6 8. Qd2 Qxb2), either side can lose the queen or get mated in one careless tempo.",
    ),
    OpeningEntry(
        eco="B70",
        name="Sicilian Defense: Dragon Variation",
        moves=("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "g6"),
        white_plan="Launch the Yugoslav Attack (Be3, f3, Qd2, O-O-O, h4-h5) to pry open the h-file and trade off the g7 bishop.",
        black_plan="Fianchetto the dark-squared bishop on g7, castle kingside, and counterattack on the c-file with ...Rc8 and ...Rxc3 sacrifices.",
        common_trap="If Black plays ...O-O carelessly without stopping h4-h5 and Bh6, White's battery mates rapidly along the h-file.",
    ),
    OpeningEntry(
        eco="B22",
        name="Sicilian Defense: Alapin Variation",
        moves=("e4", "c5", "c3"),
        white_plan="Establish a strong pawn duo on d4 and e4 with 3. d4 while sidestepping sharp Open Sicilian theory.",
        black_plan="Strike back immediately with 2...Nf6 3. e5 Nd5 or 2...d5 3. exd5 Qxd5 to prevent an uncontested White center.",
        common_trap="After 2...d5 3. exd5 Qxd5 4. d4 Nf6 5. Nf3 Nc6, playing ...Bg4 too early can leave b7 hanging to Qb3.",
    ),
    OpeningEntry(
        eco="B23",
        name="Sicilian Defense: Closed Variation",
        moves=("e4", "c5", "Nc3"),
        white_plan="Fianchetto with g3 and Bg2, play d3, and launch a kingside pawn storm with f4-f5.",
        black_plan="Develop with ...Nc6, ...g6, ...Bg7, and expand on the queenside with ...Rb8 and ...b5-b4.",
        common_trap="Ignoring Black's ...Nd4 outpost and queenside push lets Black seize the long diagonal and crush White's queenside.",
    ),
    OpeningEntry(
        eco="C00",
        name="French Defense",
        moves=("e4", "e6"),
        white_plan="Claim central space with 2. d4 d5 and choose between Nc3, Nd2, e5, or exd5.",
        black_plan="Undermine White's central chain with ...c5 and ...f6 while solving the light-squared bishop on c8.",
        common_trap="Prematurely opening the center before castling leaves Black vulnerable to Bxh7+ Greek Gift sacrifices.",
    ),
    OpeningEntry(
        eco="C02",
        name="French Defense: Advance Variation",
        moves=("e4", "e6", "d4", "d5", "e5"),
        white_plan="Maintain the d4-e5 pawn chain with c3 and Nf3 and leverage the extra kingside space to attack.",
        black_plan="Attack the base of the pawn chain with ...c5, ...Nc6, and ...Qb6, piling pressure onto d4.",
        common_trap="In the Milner-Barry trap (3...c5 4. c3 Nc6 5. Nf3 Qb6 6. Bd3 cxd4 7. cxd4 Nxd4?? 8. Nxd4 Qxd4??), 9. Bb5+ wins Black's queen!",
    ),
    OpeningEntry(
        eco="C15",
        name="French Defense: Winawer Variation",
        moves=("e4", "e6", "d4", "d5", "Nc3", "Bb4"),
        white_plan="Push 4. e5 and exploit the dark squares and bishop pair after ...Bxc3+ bxc3, targeting g7 with Qg4.",
        black_plan="Damage White's queenside structure with ...Bxc3+, counterattack d4 with ...c5, and target the c4/c3 weaknesses.",
        common_trap="After 4. e5 c5 5. a3 Bxc3+ 6. bxc3 Ne7 7. Qg4, grabbing g2 or castling blindly walks into devastating kingside attacks.",
    ),
    OpeningEntry(
        eco="B10",
        name="Caro-Kann Defense",
        moves=("e4", "c6"),
        white_plan="Take the center with 2. d4 d5 and challenge Black with the Advance, Classical, Exchange, or Panov Attack.",
        black_plan="Develop the light-squared bishop outside the pawn chain before playing ...e6, aiming for a rock-solid endgame structure.",
        common_trap="In the Two Knights / Classical lines, 1. e4 c6 2. d4 d5 3. Nc3 dxe4 4. Nxe4 Nd7 5. Qe2 Ngf6?? runs straight into 6. Nd6# smothered mate!",
    ),
    OpeningEntry(
        eco="B18",
        name="Caro-Kann Defense: Classical Variation",
        moves=("e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Bf5"),
        white_plan="Chase the bishop with 5. Ng3 Bg6 6. h4 h6 7. Nf3 and seek a space and development edge.",
        black_plan="Consolidate with ...Nd7, ...e6, and ...Ngf6, trading into a structurally sound middlegame and endgame.",
        common_trap="Forgetting to create a retreat square with ...h6 after 6. h4 allows 7. h5 to trap Black's light-squared bishop on g6.",
    ),
    OpeningEntry(
        eco="B12",
        name="Caro-Kann Defense: Advance Variation",
        moves=("e4", "c6", "d4", "d5", "e5"),
        white_plan="Preserve the space advantage with Nf3 and Be2 (Short System) or attack sharply with h4 (Tal Variation).",
        black_plan="Develop the c8 bishop to f5 before ...e6 and undermine the center with ...c5.",
        common_trap="In the Tal Variation (3...Bf5 4. h4), playing 4...e6? walks into 5. g4 Be4 6. f3 Bg6 7. h5, trapping the bishop.",
    ),
    OpeningEntry(
        eco="B01",
        name="Scandinavian Defense",
        moves=("e4", "d5"),
        white_plan="Capture 2. exd5 and gain development tempos with Nc3 against Black's early queen sortie.",
        black_plan="Recapture with ...Qxd5 (retreating to a5 or d6) or play 2...Nf6, aiming for a sturdy pawn structure with ...c6 and ...Bf5.",
        common_trap="After 2. exd5 Qxd5 3. Nc3 Qa5 4. d4 c6 5. Nf3 Bg4, carelessly leaving d5 or b7 undefended invites tactical shots like Ne5.",
    ),
    OpeningEntry(
        eco="B07",
        name="Pirc Defense",
        moves=("e4", "d6", "d4", "Nf6", "Nc3", "g6"),
        white_plan="Use the broad pawn center with the Austrian Attack (f4) or Classical setup (Nf3, Be2) to squeeze Black.",
        black_plan="Complete kingside fianchetto with ...Bg7 and ...O-O, then strike White's center with ...c5 or ...e5.",
        common_trap="Allowing White's e4-e5 break before castling can displace the f6 knight and expose Black's king on the e-file.",
    ),
    OpeningEntry(
        eco="B02",
        name="Alekhine's Defense",
        moves=("e4", "Nf6"),
        white_plan="Claim space with 2. e5 Nd5 3. d4 d6 4. Nf3 (or 4. c4 Nb6 5. f4) without overextending the pawn front.",
        black_plan="Provoke White's central pawns forward so they become targets for ...d6, ...c5, and piece pressure.",
        common_trap="Overpushing with 2. e5 Nd5 3. c4 Nb6 4. c5?! Nd5 5. Bc4 gives Black easy targets if White cannot support the advanced pawns.",
    ),
    # ------------------------------------------------------------------
    # 1.d4 — Closed and Indian Defenses
    # ------------------------------------------------------------------
    OpeningEntry(
        eco="D06",
        name="Queen's Gambit",
        moves=("d4", "d5", "c4"),
        white_plan="Challenge Black's d5 pawn to win central space and open the c-file for queenside pressure.",
        black_plan="Maintain the central anchor with ...e6 (QGD) or ...c6 (Slav), or accept and seek active piece play.",
        common_trap="In the QGA, trying to cling to the c4 pawn with ...b5? fails to a4! c6 axb5 cxb5 Qf3!, winning the a8 rook.",
    ),
    OpeningEntry(
        eco="D30",
        name="Queen's Gambit Declined",
        moves=("d4", "d5", "c4", "e6"),
        white_plan="Develop with Nc3, Bg5 (pinning the f6 knight), and either play a minority attack on the queenside or push e4.",
        black_plan="Keep d5 rock solid, castle early, and free the c8 bishop via ...c5 or ...e5 breaks.",
        common_trap="The Elephant Trap: after 3. Nc3 Nf6 4. Bg5 Nbd7 5. cxd5 exd5 6. Nxd5? Nxd5! 7. Bxd8 Bb4+ wins a full piece for Black.",
    ),
    OpeningEntry(
        eco="D20",
        name="Queen's Gambit Accepted",
        moves=("d4", "d5", "c4", "dxc4"),
        white_plan="Regain the c4 pawn smoothly with Nf3, e3, and Bxc4 while enjoying an uncontested central majority.",
        black_plan="Give back the pawn at the right moment and strike White's center with ...c5 and ...a6/...b5.",
        common_trap="After 3. e3 b5? 4. a4 c6 5. axb5 cxb5 6. Qf3!, Black's a8 rook is trapped on the long diagonal.",
    ),
    OpeningEntry(
        eco="D10",
        name="Slav Defense",
        moves=("d4", "d5", "c4", "c6"),
        white_plan="Develop with Nf3 and Nc3, aiming to punish Black's slightly passive queenside or seize the center with e4.",
        black_plan="Support d5 without blocking the c8 bishop, developing it to f5 or g4 before playing ...e6.",
        common_trap="After 3. Nf3 Nf6 4. Nc3 dxc4 5. a4 Bf5, playing 6. Ne5 Nbd7 7. Nxc4 Qc7 requires care to avoid ...e5 counterstrikes.",
    ),
    OpeningEntry(
        eco="D02",
        name="London System",
        moves=("d4", "d5", "Nf3", "Nf6", "Bf4"),
        white_plan="Form a resilient pyramid with e3, c3, Bd3, and Nbd2, then plant a knight on e5 for a kingside attack.",
        black_plan="Challenge White's structure early with ...c5 and ...Qb6 (targeting b2) or ...Bd6 to trade off the London bishop.",
        common_trap="Playing on autopilot without noticing ...Qb6 drops the b2 pawn or allows ...cxd4 cxd4 Qb6 tactics.",
    ),
    OpeningEntry(
        eco="E60",
        name="King's Indian Defense",
        moves=("d4", "Nf6", "c4", "g6", "Nc3", "Bg7"),
        white_plan="Build a massive center with e4, castle kingside, and crash through on the queenside with c5.",
        black_plan="Castle early, strike the center with ...e5 or ...c5, and launch a fierce kingside pawn storm with ...f5-f4.",
        common_trap="In the Mar del Plata line, ignoring Black's ...f5-f4 and ...g5-g4 kingside avalanche leads to a mating attack even when White wins on the queenside.",
    ),
    OpeningEntry(
        eco="E20",
        name="Nimzo-Indian Defense",
        moves=("d4", "Nf6", "c4", "e6", "Nc3", "Bb4"),
        white_plan="Secure the bishop pair and central pawns via Qc2, e3, or a3, aiming to open the position later.",
        black_plan="Control the e4 square, double White's c-pawns with ...Bxc3+, and blockade the c4 square with ...c5 and ...Nc6-a5.",
        common_trap="Greedily chasing the bishop without defending e4 lets Black dominate the dark squares and blockade c4 permanently.",
    ),
    OpeningEntry(
        eco="E12",
        name="Queen's Indian Defense",
        moves=("d4", "Nf6", "c4", "e6", "Nf3", "b6"),
        white_plan="Fianchetto with g3 and Bg2 to neutralize Black's b7 bishop and fight for the d5 and e4 squares.",
        black_plan="Control e4 from afar with ...Bb7 and ...Bb4, maintaining a flexible, hypermodern pawn structure.",
        common_trap="Careless central pushes by White can hang the d4 pawn or allow ...Bb4+ tactics disrupting White's coordination.",
    ),
    OpeningEntry(
        eco="D80",
        name="Grunfeld Defense",
        moves=("d4", "Nf6", "c4", "g6", "Nc3", "d5"),
        white_plan="Capture 4. cxd5 Nxd5 5. e4 Nxc3 6. bxc3 to erect a dominant pawn center and push d5.",
        black_plan="Put relentless piece and pawn pressure on White's d4 center with ...Bg7, ...c5, and ...Nc6.",
        common_trap="If White forgets to defend d4/a1 along the h8-a1 diagonal, ...Bg7 and ...c5 shatter the center and win material.",
    ),
    OpeningEntry(
        eco="A80",
        name="Dutch Defense",
        moves=("d4", "f5"),
        white_plan="Fianchetto on g2, control e4, and exploit the weakened e8-h5 diagonal and light squares.",
        black_plan="Fight for e4 and build a Stonewall (...d5, ...e6, ...c6) or Leningrad (...g6, ...Bg7) kingside attack.",
        common_trap="After 2. Bg5 h6 3. Bh4 g5 4. e3! (threatening Qh5#), capturing 4...gxh4?? walks into immediate checkmate with 5. Qh5#.",
    ),
    OpeningEntry(
        eco="A60",
        name="Benoni Defense",
        moves=("d4", "Nf6", "c4", "c5", "d5", "e6"),
        white_plan="Play Nc3 and e4 to dominate the center and prepare an e4-e5 central breakthrough.",
        black_plan="Create an asymmetrical queenside pawn majority after ...exd5 cxd5 d6 and attack with ...b5 and the g7 bishop.",
        common_trap="Allowing White's e4-e5 break when Black's pieces are uncoordinated rips open the e-file against Black's king.",
    ),
    OpeningEntry(
        eco="A40",
        name="Englund Gambit Trap",
        moves=("d4", "e5", "dxe5", "Nc6", "Nf3", "Qe7"),
        white_plan="Develop calmly with Bf4 or Nc3, return the extra pawn if needed for rapid development, and exploit Black's awkward queen.",
        black_plan="Pressure e5 with ...Qe7 and ...Qb4+ to win the b2 pawn and set tactical traps against White's queenside.",
        common_trap="After 4. Bf4 Qb4+ 5. Bd2 Qxb2 6. Bc3?? Bb4 7. Qd2 Bxc3 8. Qxc3 Qc1# is a notorious back-rank checkmate.",
    ),
    # ------------------------------------------------------------------
    # Flank Openings
    # ------------------------------------------------------------------
    OpeningEntry(
        eco="A10",
        name="English Opening",
        moves=("c4",),
        white_plan="Control d5 from the flank, fianchetto with g3 and Bg2, and steer into favorable reversed Sicilian or d4 structures.",
        black_plan="Claim central space with 1...e5 (Reversed Sicilian), mirror with 1...c5 (Symmetrical), or transpose into QGD/Slav setups.",
        common_trap="In the Reversed Dragon, carelessly leaving the long diagonal undefended lets White's g2 bishop win the a8 rook or c6 knight.",
    ),
    OpeningEntry(
        eco="A09",
        name="Reti Opening",
        moves=("Nf3", "d5", "c4"),
        white_plan="Undermine Black's d5 center from the flank while keeping d2-d4 flexible and fianchettoing with g3 and Bg2.",
        black_plan="Choose between 2...d4 (gaining space), 2...e6/c6 (holding d5), or 2...dxc4 (seeking open piece play).",
        common_trap="After 2...d4 3. e3 Nc6 4. exd4 Nxd4 5. Nxd4 Qxd4, overextending without development leaves Black's queen exposed to Nc3 and d3.",
    ),
    OpeningEntry(
        eco="A02",
        name="Bird's Opening",
        moves=("f4",),
        white_plan="Clamp down on e5, develop with Nf3, e3, b3, and Bb2, and aim for a reversed Dutch kingside attack.",
        black_plan="Counter solidly with 1...d5 and ...g6/...Bg7, or play the sharp From's Gambit (1...e5) targeting White's weakened kingside.",
        common_trap="In From's Gambit (1...e5 2. fxe5 d6 3. exd6 Bxd6), playing 4. Nf3? allows 4...g5! and ...g4-Qh4+ mating threats along the e1-h4 diagonal.",
    ),
)


def _canonical_board_fen(fen: str) -> str:
    """Canonicalize a 4- or 6-field FEN to its normalized 4-field position key."""
    board = chess.Board(fen.strip())
    return normalize_board_fen(board.fen())


@lru_cache(maxsize=1)
def _fen_index() -> dict[str, OpeningEntry]:
    """Replay every entry in ``OPENING_BOOK`` and index by normalized 4-field FEN."""
    index: dict[str, OpeningEntry] = {}
    for entry in OPENING_BOOK:
        board = chess.Board()
        for san in entry.moves:
            board.push_san(san)
        key = normalize_board_fen(board.fen())
        if key in index:
            raise ValueError(
                f"Duplicate opening FEN key for {entry.name!r} and {index[key].name!r}: {key}"
            )
        index[key] = entry
    return index


def identify_opening(
    fen: str | None = None,
    moves_san: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """Identify an opening by board FEN and/or a SAN move history from the start.

    When ``moves_san`` is provided, walks the game from ``chess.STARTING_FEN``
    and returns the deepest matching opening in ``OPENING_BOOK`` (automatically
    transposition-aware because every ply is matched via its canonical 4-field
    board FEN). If ``moves_san`` is omitted or yields no match and ``fen`` is
    provided, falls back to direct 4-field FEN lookup.
    """
    index = _fen_index()

    if moves_san:
        board = chess.Board()
        deepest_match: OpeningEntry | None = None
        for san in moves_san:
            try:
                board.push_san(san)
            except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
                break
            hit = index.get(normalize_board_fen(board.fen()))
            if hit is not None:
                deepest_match = hit
        if deepest_match is not None:
            return deepest_match.to_dict()

    if fen is not None:
        try:
            key = _canonical_board_fen(fen)
        except ValueError:
            return None
        hit = index.get(key)
        if hit is not None:
            return hit.to_dict()

    return None


def list_openings(eco_prefix: str | None = None) -> list[dict[str, Any]]:
    """Return all openings in ``OPENING_BOOK``, optionally filtered by ECO prefix."""
    if not eco_prefix:
        return [entry.to_dict() for entry in OPENING_BOOK]
    prefix = eco_prefix.strip().upper()
    return [entry.to_dict() for entry in OPENING_BOOK if entry.eco.upper().startswith(prefix)]
