"""Unit tests for Lichess/Chess.com game importers and multi-game player profiling."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from chessfm.coach.importers import (
    DEFAULT_USER_AGENT,
    ImporterError,
    detect_student_color,
    fetch_chesscom_games,
    fetch_lichess_games,
    format_profile_report,
    main,
    profile_player_games,
    split_pgn_games,
)

GAME_1_PGN = """\
[Event "Rated Blitz game"]
[Site "https://lichess.org/abc12345"]
[White "Alice_FM"]
[Black "OpponentOne"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 1-0
"""

GAME_2_PGN = """\
[Event "Rated Rapid game"]
[Site "https://lichess.org/def67890"]
[White "OpponentTwo"]
[Black "alice_fm"]
[Result "1-0"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0
"""

GAME_3_PGN = """\
[Event "Casual game"]
[White "PlayerA"]
[Black "PlayerB"]
[Result "1/2-1/2"]

1. d4 d5 2. c4 e6 1/2-1/2
"""


def _mock_response(
    *,
    status_code: int = 200,
    text: str = "",
    json_data: object | None = None,
    json_error: bool = False,
) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = text
    if json_error:
        resp.json.side_effect = ValueError("Invalid JSON")
    else:
        resp.json.return_value = json_data
    return resp


class TestSplitPgnGames:
    """Tests for multi-game PGN bundle splitting."""

    def test_splits_multiple_games_cleanly(self) -> None:
        bundle = f"{GAME_1_PGN}\n\n{GAME_2_PGN}\n\n{GAME_3_PGN}"
        games = split_pgn_games(bundle)
        assert len(games) == 3
        assert "Alice_FM" in games[0]
        assert "Alice_FM" not in games[1]
        assert "OpponentTwo" in games[1]
        assert "OpponentTwo" not in games[2]
        assert "PlayerA" in games[2]

    def test_respects_max_games(self) -> None:
        bundle = f"{GAME_1_PGN}\n\n{GAME_2_PGN}\n\n{GAME_3_PGN}"
        games = split_pgn_games(bundle, max_games=2)
        assert len(games) == 2
        assert "Alice_FM" in games[0]
        assert "OpponentTwo" in games[1]

    def test_max_games_zero_or_negative_returns_empty(self) -> None:
        assert split_pgn_games(GAME_1_PGN, max_games=0) == []
        assert split_pgn_games(GAME_1_PGN, max_games=-3) == []

    def test_empty_or_whitespace_returns_empty(self) -> None:
        assert split_pgn_games("") == []
        assert split_pgn_games("   \n\t  ") == []


class TestDetectStudentColor:
    """Tests for case-insensitive PGN header color detection."""

    def test_detects_white_case_insensitive(self) -> None:
        assert detect_student_color(GAME_1_PGN, "alice_fm") == "white"
        assert detect_student_color(GAME_1_PGN, "  ALICE_FM ") == "white"

    def test_detects_black_case_insensitive(self) -> None:
        assert detect_student_color(GAME_2_PGN, "ALICE_FM") == "black"
        assert detect_student_color(GAME_2_PGN, "alice_fm") == "black"

    def test_defaults_to_white_when_not_matched_or_blank(self) -> None:
        assert detect_student_color(GAME_3_PGN, "alice_fm") == "white"
        assert detect_student_color(GAME_2_PGN, "") == "white"
        assert detect_student_color("", "alice_fm") == "white"
        assert detect_student_color("1. e4 e5 2. Nf3 Nc6", "alice_fm") == "white"


class TestFetchLichessGames:
    """Tests for Lichess public API game fetching."""

    @patch("chessfm.coach.importers.requests.get")
    def test_fetches_and_splits_pgns(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response(
            status_code=200,
            text=f"{GAME_1_PGN}\n\n{GAME_2_PGN}",
        )
        pgns = fetch_lichess_games("Alice_FM", max_games=5, timeout=12)
        assert len(pgns) == 2
        assert "Alice_FM" in pgns[0]

        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == (
            "https://lichess.org/api/games/user/Alice_FM?max=5&evals=false&opening=true"
        )
        assert kwargs["headers"]["User-Agent"] == DEFAULT_USER_AGENT
        assert kwargs["headers"]["Accept"] == "application/x-chess-pgn"
        assert kwargs["timeout"] == 12

    @patch("chessfm.coach.importers.requests.get")
    def test_zero_max_games_skips_network(self, mock_get: MagicMock) -> None:
        assert fetch_lichess_games("Alice_FM", max_games=0) == []
        mock_get.assert_not_called()

    def test_blank_username_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            fetch_lichess_games("   ")

    @pytest.mark.parametrize(
        ("status_code", "expected_fragment"),
        [
            (404, "not found"),
            (429, "rate limit"),
            (503, "HTTP 503"),
        ],
    )
    @patch("chessfm.coach.importers.requests.get")
    def test_http_errors_raise_importer_error(
        self, mock_get: MagicMock, status_code: int, expected_fragment: str
    ) -> None:
        mock_get.return_value = _mock_response(status_code=status_code)
        with pytest.raises(ImporterError, match=expected_fragment):
            fetch_lichess_games("ghost_user", max_games=3)

    @patch("chessfm.coach.importers.requests.get")
    def test_network_exception_raises_importer_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = requests.Timeout("timed out")
        with pytest.raises(ImporterError, match="Failed to fetch Lichess games"):
            fetch_lichess_games("Alice_FM", max_games=3)


class TestFetchChesscomGames:
    """Tests for Chess.com public monthly archives API fetching."""

    @patch("chessfm.coach.importers.requests.get")
    def test_fetches_newest_archives_first_and_respects_limit(self, mock_get: MagicMock) -> None:
        archives_resp = _mock_response(
            status_code=200,
            json_data={
                "archives": [
                    "https://api.chess.com/pub/player/alice_fm/games/2026/01",
                    "https://api.chess.com/pub/player/alice_fm/games/2026/02",
                ]
            },
        )
        feb_resp = _mock_response(
            status_code=200,
            json_data={
                "games": [
                    {"pgn": GAME_1_PGN},
                    {"no_pgn": True},  # skipped
                    "not-a-dict",  # skipped
                    {"pgn": GAME_2_PGN},
                ]
            },
        )
        jan_resp = _mock_response(
            status_code=200,
            json_data={"games": [{"pgn": GAME_3_PGN}]},
        )
        mock_get.side_effect = [archives_resp, feb_resp, jan_resp]

        pgns = fetch_chesscom_games("Alice_FM", max_games=3, timeout=10)
        assert len(pgns) == 3
        # Newest archive (Feb) reversed first -> GAME_2, GAME_1, then Jan -> GAME_3
        assert "OpponentTwo" in pgns[0]
        assert "Alice_FM" in pgns[1]
        assert "PlayerA" in pgns[2]

        first_url = mock_get.call_args_list[0].args[0]
        assert first_url == "https://api.chess.com/pub/player/alice_fm/games/archives"
        assert mock_get.call_args_list[0].kwargs["headers"]["User-Agent"] == DEFAULT_USER_AGENT

    @patch("chessfm.coach.importers.requests.get")
    def test_empty_archives_returns_empty(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response(status_code=200, json_data={"archives": []})
        assert fetch_chesscom_games("alice_fm", max_games=5) == []

    @patch("chessfm.coach.importers.requests.get")
    def test_skips_invalid_month_entries_and_returns_available(self, mock_get: MagicMock) -> None:
        archives_resp = _mock_response(
            status_code=200,
            json_data={
                "archives": [
                    "",
                    "https://api.chess.com/pub/player/alice_fm/games/2026/02",
                ]
            },
        )
        feb_resp = _mock_response(status_code=200, json_data={"games": "not-a-list"})
        mock_get.side_effect = [archives_resp, feb_resp]
        assert fetch_chesscom_games("alice_fm", max_games=5) == []

    @patch("chessfm.coach.importers.requests.get")
    def test_zero_max_games_skips_network(self, mock_get: MagicMock) -> None:
        assert fetch_chesscom_games("alice_fm", max_games=0) == []
        mock_get.assert_not_called()

    def test_blank_username_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            fetch_chesscom_games("")

    @patch("chessfm.coach.importers.requests.get")
    def test_archives_errors(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = requests.ConnectionError("dns failure")
        with pytest.raises(ImporterError, match="Failed to fetch Chess.com archives"):
            fetch_chesscom_games("alice_fm")

        mock_get.side_effect = None
        mock_get.return_value = _mock_response(status_code=404)
        with pytest.raises(ImporterError, match="not found"):
            fetch_chesscom_games("alice_fm")

        mock_get.return_value = _mock_response(status_code=200, json_error=True)
        with pytest.raises(ImporterError, match="Invalid JSON in Chess.com archives"):
            fetch_chesscom_games("alice_fm")

    @patch("chessfm.coach.importers.requests.get")
    def test_monthly_archive_errors(self, mock_get: MagicMock) -> None:
        archives_ok = _mock_response(
            status_code=200,
            json_data={"archives": ["https://api.chess.com/pub/player/alice_fm/games/2026/02"]},
        )
        # Network exception on monthly fetch
        mock_get.side_effect = [archives_ok, requests.Timeout("slow")]
        with pytest.raises(ImporterError, match="Failed to fetch Chess.com archive"):
            fetch_chesscom_games("alice_fm")

        # HTTP 500 on monthly fetch
        mock_get.side_effect = [archives_ok, _mock_response(status_code=500)]
        with pytest.raises(ImporterError, match="HTTP 500"):
            fetch_chesscom_games("alice_fm")

        # Invalid JSON on monthly fetch
        mock_get.side_effect = [archives_ok, _mock_response(status_code=200, json_error=True)]
        with pytest.raises(ImporterError, match="Invalid JSON in Chess.com monthly archive"):
            fetch_chesscom_games("alice_fm")


class TestProfilePlayerGames:
    """Tests for batch game review and weakness profile aggregation."""

    def test_profiles_multiple_games_with_auto_color_detection(self) -> None:
        result = profile_player_games(
            [GAME_1_PGN, "   ", GAME_2_PGN],
            username="alice_fm",
            engine=None,
            student_elo=1400,
            depth=4,
        )
        assert set(result.keys()) == {
            "username",
            "games_analyzed",
            "mean_accuracy",
            "reviews",
            "weakness_profile",
        }
        assert result["username"] == "alice_fm"
        assert result["games_analyzed"] == 2
        assert len(result["reviews"]) == 2
        assert result["reviews"][0]["student_color"] == "white"
        assert result["reviews"][1]["student_color"] == "black"
        assert result["reviews"][0]["student_elo"] == 1400
        assert 0.0 <= result["mean_accuracy"] <= 100.0
        assert result["weakness_profile"]["games_analyzed"] == 2

    def test_empty_pgns_returns_zero_profile(self) -> None:
        result = profile_player_games([], username="alice_fm", engine=None)
        assert result["games_analyzed"] == 0
        assert result["mean_accuracy"] == 0.0
        assert result["reviews"] == []
        assert result["weakness_profile"]["games_analyzed"] == 0

    def test_format_profile_report_with_and_without_priorities(self) -> None:
        empty_report = format_profile_report(profile_player_games([], username="", engine=None))
        assert "Student" in empty_report
        assert "none detected" in empty_report

        populated = profile_player_games([GAME_2_PGN], username="alice_fm", engine=None)
        text = format_profile_report(populated)
        assert "alice_fm" in text
        assert "Games Analyzed: 1" in text
        assert "Practice Drills Extracted:" in text


class TestImporterCLI:
    """Tests for the ``python -m chessfm.coach.importers`` CLI."""

    def test_cli_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "chessfm.coach.importers", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0
        assert "--platform" in proc.stdout
        assert "--pgn-file" in proc.stdout
        assert "--json" in proc.stdout

    def test_cli_pgn_file_json_and_text(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pgn_path = tmp_path / "games.pgn"
        pgn_path.write_text(f"{GAME_1_PGN}\n\n{GAME_2_PGN}", encoding="utf-8")

        rc = main(
            [
                "--pgn-file",
                str(pgn_path),
                "--username",
                "alice_fm",
                "--games",
                "2",
                "--no-engine",
                "--json",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["username"] == "alice_fm"
        assert payload["games_analyzed"] == 2
        assert payload["reviews"][0]["student_color"] == "white"
        assert payload["reviews"][1]["student_color"] == "black"

        rc_text = main(
            [
                "--pgn-file",
                str(pgn_path),
                "--username",
                "alice_fm",
                "--games",
                "1",
                "--no-engine",
            ]
        )
        assert rc_text == 0
        out = capsys.readouterr().out
        assert "ChessFM Player Profile" in out
        assert "Games Analyzed: 1" in out

    def test_cli_missing_pgn_file_returns_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["--pgn-file", str(tmp_path / "missing.pgn")])
        assert rc == 1
        assert "PGN file not found" in capsys.readouterr().err

    def test_cli_validation_errors(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main([]) == 2
        assert "specify --platform" in capsys.readouterr().err

        assert main(["--platform", "lichess"]) == 2
        assert "--username is required" in capsys.readouterr().err

    @patch("chessfm.coach.importers.close_engine")
    @patch("chessfm.coach.importers.open_engine", return_value=None)
    @patch("chessfm.coach.importers.fetch_lichess_games", return_value=[GAME_1_PGN])
    def test_cli_remote_lichess(
        self,
        mock_fetch: MagicMock,
        mock_open: MagicMock,
        mock_close: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main(["--platform", "lichess", "--username", "Alice_FM", "--games", "1", "--json"])
        assert rc == 0
        mock_fetch.assert_called_once_with("Alice_FM", max_games=1, timeout=15)
        mock_open.assert_called_once()
        mock_close.assert_called_once_with(None)
        payload = json.loads(capsys.readouterr().out)
        assert payload["games_analyzed"] == 1

    @patch("chessfm.coach.importers.fetch_chesscom_games", return_value=[GAME_2_PGN])
    def test_cli_remote_chesscom(
        self, mock_fetch: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(
            [
                "--platform",
                "chesscom",
                "--username",
                "alice_fm",
                "--games",
                "1",
                "--no-engine",
                "--json",
            ]
        )
        assert rc == 0
        mock_fetch.assert_called_once_with("alice_fm", max_games=1, timeout=15)
        payload = json.loads(capsys.readouterr().out)
        assert payload["games_analyzed"] == 1
        assert payload["reviews"][0]["student_color"] == "black"

    @patch(
        "chessfm.coach.importers.fetch_lichess_games",
        side_effect=ImporterError("Lichess user 'nope' not found (HTTP 404)."),
    )
    def test_cli_remote_importer_error(
        self, _mock_fetch: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["--platform", "lichess", "--username", "nope", "--no-engine"])
        assert rc == 1
        assert "HTTP 404" in capsys.readouterr().err
