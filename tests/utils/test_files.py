import os
from pathlib import Path

import pytest

from oceanicospy.utils.files import (
    count_lines,
    count_NGRID_occurrences,
    create_link,
    delete_line,
    deploy_input_file,
    duplicate_lines,
    fill_files,
    fill_files_only_once,
    look_for_NGRID_linenumber,
    remove_link,
    verify_file,
    verify_link,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def template_file(tmp_path):
    f = tmp_path / "template.txt"
    f.write_text("Hello $name, your value is $value.\n")
    return f


@pytest.fixture
def ngrid_file(tmp_path):
    f = tmp_path / "grid.swn"
    f.write_text("PREAMBLE\nNGRID first params\nNGRID second params\nEOF\n")
    return f


# ---------------------------------------------------------------------------
# fill_files
# ---------------------------------------------------------------------------


class TestFillFiles:
    def test_replaces_known_placeholders(self, template_file):
        fill_files(str(template_file), {"name": "World", "value": "42"})
        content = template_file.read_text()
        assert "World" in content
        assert "42" in content
        assert "$name" not in content
        assert "$value" not in content

    def test_empty_value_becomes_whitespace(self, template_file):
        fill_files(str(template_file), {"name": "", "value": "X"})
        content = template_file.read_text()
        assert "$name" not in content

    def test_unknown_key_left_untouched(self, template_file):
        fill_files(str(template_file), {"name": "Alice"})
        content = template_file.read_text()
        assert "$value" in content


# ---------------------------------------------------------------------------
# fill_files_only_once
# ---------------------------------------------------------------------------


class TestFillFilesOnlyOnce:
    def test_replaces_first_occurrence_only(self, tmp_path):
        f = tmp_path / "once.txt"
        f.write_text("$key appears $key twice\n")
        fill_files_only_once(str(f), {"key": "VALUE"})
        content = f.read_text()
        assert content.count("VALUE") == 1
        assert "$key" in content  # second occurrence unchanged

    def test_nest_id_replaced_twice(self, tmp_path):
        f = tmp_path / "nest.txt"
        f.write_text("$nest_id first $nest_id second $nest_id third\n")
        fill_files_only_once(str(f), {"nest_id": "N1"})
        content = f.read_text()
        assert content.count("N1") == 2
        assert "$nest_id" in content  # third occurrence unchanged


# ---------------------------------------------------------------------------
# NGRID helpers
# ---------------------------------------------------------------------------


class TestNGRIDHelpers:
    def test_look_for_linenumber_returns_correct_line(self, ngrid_file):
        assert look_for_NGRID_linenumber(str(ngrid_file)) == 2

    def test_look_for_linenumber_no_ngrid(self, tmp_path):
        f = tmp_path / "empty.swn"
        f.write_text("no grid here\n")
        assert look_for_NGRID_linenumber(str(f)) is False

    def test_count_ngrid_occurrences(self, ngrid_file):
        assert count_NGRID_occurrences(str(ngrid_file)) == 2

    def test_count_ngrid_zero(self, tmp_path):
        f = tmp_path / "zero.swn"
        f.write_text("nothing here\n")
        assert count_NGRID_occurrences(str(f)) == 0


# ---------------------------------------------------------------------------
# count_lines / delete_line / duplicate_lines
# ---------------------------------------------------------------------------


class TestLineOperations:
    def test_count_lines(self, tmp_path):
        f = tmp_path / "lines.txt"
        f.write_text("a\nb\nc\n")
        assert count_lines(str(f)) == 3

    def test_count_lines_single(self, tmp_path):
        f = tmp_path / "one.txt"
        f.write_text("only one line\n")
        assert count_lines(str(f)) == 1

    def test_delete_line_removes_matching(self, tmp_path):
        f = tmp_path / "del.txt"
        f.write_text("keep this\ndelete me\nalso keep\n")
        delete_line(str(f), "me")
        lines = f.read_text().splitlines()
        assert len(lines) == 2
        assert all("delete" not in l for l in lines)

    def test_delete_line_no_match_unchanged(self, tmp_path):
        f = tmp_path / "unchanged.txt"
        f.write_text("line1\nline2\n")
        delete_line(str(f), "nothere")
        assert count_lines(str(f)) == 2

    def test_duplicate_lines_increases_count(self, tmp_path):
        f = tmp_path / "dup.txt"
        f.write_text("line1\nline2\nline3\n")
        duplicate_lines(str(f), 1)
        # lines 1-2 duplicated → 5 lines total
        assert count_lines(str(f)) == 5

    def test_duplicate_lines_invalid_index_raises(self, tmp_path):
        f = tmp_path / "short.txt"
        f.write_text("only\none\n")
        with pytest.raises(IndexError):
            duplicate_lines(str(f), 2)  # idx+1 = 2, len=2 → 2 >= 2 triggers error


# ---------------------------------------------------------------------------
# verify_file / verify_link / create_link / remove_link
# ---------------------------------------------------------------------------


class TestLinkOperations:
    def test_verify_file_existing(self, tmp_path):
        f = tmp_path / "real.txt"
        f.write_text("data")
        assert verify_file(str(f)) is True

    def test_verify_file_missing(self, tmp_path):
        assert verify_file(str(tmp_path / "ghost.txt")) is False

    def test_verify_link_no_link(self, tmp_path):
        assert verify_link("nolink.txt", str(tmp_path) + "/") is False

    def test_create_and_remove_link(self, tmp_path):
        src = tmp_path / "source.txt"
        src.write_text("payload")
        link_dir = tmp_path / "links"
        link_dir.mkdir()

        create_link("source.txt", str(tmp_path) + "/", str(link_dir) + "/")
        assert os.path.islink(str(link_dir / "source.txt"))

        remove_link("source.txt", str(link_dir))
        assert not os.path.islink(str(link_dir / "source.txt"))

    def test_create_link_skips_if_file_exists(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("data")
        dst = tmp_path / "dst" / "src.txt"
        dst.parent.mkdir()
        dst.write_text("already here")

        # Should return None and not raise
        result = create_link("src.txt", str(tmp_path) + "/", str(dst.parent) + "/")
        assert result is None


# ---------------------------------------------------------------------------
# deploy_input_file
# ---------------------------------------------------------------------------


class TestDeployInputFile:
    def test_none_use_link_is_noop(self, tmp_path):
        src = tmp_path / "origin" / "file.dat"
        src.parent.mkdir()
        src.write_text("data")
        run = tmp_path / "run"
        run.mkdir()

        deploy_input_file("file.dat", str(src.parent) + "/", str(run) + "/", use_link=None)
        assert not (run / "file.dat").exists()

    def test_copy_mode(self, tmp_path):
        src = tmp_path / "origin"
        src.mkdir()
        (src / "file.dat").write_text("content")
        run = tmp_path / "run"
        run.mkdir()

        deploy_input_file("file.dat", str(src) + "/", str(run) + "/", use_link=False)
        assert (run / "file.dat").is_file()
        assert not (run / "file.dat").is_symlink()

    def test_link_mode(self, tmp_path):
        src = tmp_path / "origin"
        src.mkdir()
        (src / "file.dat").write_text("content")
        run = tmp_path / "run"
        run.mkdir()

        deploy_input_file("file.dat", str(src) + "/", str(run) + "/", use_link=True)
        assert os.path.islink(str(run / "file.dat"))
