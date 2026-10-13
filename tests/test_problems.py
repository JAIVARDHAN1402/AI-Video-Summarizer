"""The problem list a coding video leaves behind.

A three-hour SQL practice video solves seventy questions and then they are
gone. These cover the parts that decide whether the list is usable: that a
language label means one thing, that a half-extracted problem is dropped
rather than rendered, and that the windows cover the end of the video and not
just its first ninety minutes.
"""

from __future__ import annotations

import config
from core import problems


# --- language names -------------------------------------------------------
# The model writes "Postgres", "postgresql" and "PostgreSQL" for one thing.
# Left alone that is three tabs in the UI, two of which are empty.

def test_aliases_collapse_onto_one_name():
    for written in ("Postgres", "postgres", "PostgreSQL", "psql"):
        assert problems.canonical_language(written) == "PostgreSQL"
    for written in ("T-SQL", "mssql", "SQL Server", "transact-sql"):
        assert problems.canonical_language(written) == "SQL Server"


def test_an_unknown_language_is_refused_rather_than_invented():
    assert problems.canonical_language("Klingon") is None
    assert problems.canonical_language("") is None
    assert problems.canonical_language(None) is None


def test_families_do_not_mix():
    """Asking a SQL question "in Python" is a different exercise, not a
    translation, so the two sets stay apart."""
    assert problems.family_of("Oracle") == "sql"
    assert problems.family_of("C++") == "programming"
    assert set(problems.FAMILIES["sql"]).isdisjoint(problems.FAMILIES["programming"])


# --- cleaning -------------------------------------------------------------

def test_a_problem_without_a_real_solution_is_dropped():
    assert problems._clean_problem(
        {"title": "x", "statement": "Find the second highest salary", "solution": ""}
    ) is None


def test_a_problem_without_a_real_statement_is_dropped():
    assert problems._clean_problem(
        {"title": "x", "statement": "next", "solution": "SELECT 1;"}
    ) is None


def test_a_good_problem_keeps_its_solution_under_its_language():
    cleaned = problems._clean_problem({
        "title": "Second highest salary",
        "statement": "Find the second highest salary in the Employee table.",
        "solution": "SELECT MAX(salary) FROM Employee WHERE salary < (SELECT MAX(salary) FROM Employee);",
        "language": "mysql",
        "start_sec": 412,
        "difficulty": "medium",
        "topics": ["subqueries", "aggregates"],
    })
    assert cleaned["language"] == "MySQL"
    assert "MySQL" in cleaned["solutions"]
    assert cleaned["start_sec"] == 412
    assert cleaned["topics"] == ["subqueries", "aggregates"]


def test_a_fenced_solution_is_unwrapped():
    cleaned = problems._clean_problem({
        "title": "Count rows",
        "statement": "Count the rows in the patients table.",
        "solution": "```sql\nSELECT COUNT(*) FROM patients;\n```",
        "language": "MySQL",
    })
    assert cleaned["solutions"]["MySQL"] == "SELECT COUNT(*) FROM patients;"


def test_an_unrecognised_language_leaves_the_code_reachable():
    """The solution still has to be shown, even when the label was no use."""
    cleaned = problems._clean_problem({
        "title": "Something",
        "statement": "Do the thing described on the slide.",
        "solution": "print('hi')",
        "language": "Brainfuck",
    })
    assert cleaned["language"] == "Unknown"
    assert cleaned["original_solution"] == "print('hi')"


# --- the list ------------------------------------------------------------

def _p(statement, start, solution="SELECT 1;"):
    return problems._clean_problem(
        {"title": statement[:20], "statement": statement,
         "solution": solution, "language": "MySQL", "start_sec": start})


def test_the_same_question_asked_twice_appears_once():
    """Windows overlap in what they discuss, and a video revisits a question."""
    out = problems._dedupe([
        _p("Find the second highest salary in Employee.", 100),
        _p("find the SECOND highest salary in employee!", 900),
        _p("List every doctor in the hospital table.", 300),
    ])
    assert len(out) == 2


def test_the_list_is_numbered_in_the_video_s_order():
    out = problems._dedupe([
        _p("List every doctor in the hospital table.", 300),
        _p("Find the second highest salary in Employee.", 100),
    ])
    assert [p["n"] for p in out] == [1, 2]
    assert out[0]["start_sec"] == 100, "numbering follows the video, not the reply"


# --- windowing -----------------------------------------------------------

def _understanding(n_sections: int, body_chars: int = 1200):
    return {"explanation": [
        {"heading": f"Part {i}", "body": "x" * body_chars, "start_sec": i * 60}
        for i in range(n_sections)
    ]}


def test_a_short_video_is_one_window():
    assert len(problems._windows(_understanding(3), None)) == 1


def test_a_long_video_is_cut_into_several():
    pieces = problems._windows(_understanding(40), None)
    assert len(pieces) > 1
    assert all(len(p) <= problems._WINDOW_CHARS + 2000 for p in pieces)


def test_the_window_count_is_capped_so_one_video_cannot_drain_the_quota(monkeypatch):
    """Every window is a request carrying whole solutions back."""
    monkeypatch.setattr(config, "PROBLEMS_MAX_WINDOWS", 5)
    pieces = problems._windows(_understanding(200), None)
    assert len(pieces) == 5


def test_capping_keeps_the_end_of_the_video(monkeypatch):
    """Truncating would make the list stop at ninety minutes of a three-hour
    video, which is worse than covering all of it thinly."""
    monkeypatch.setattr(config, "PROBLEMS_MAX_WINDOWS", 4)
    u = {"explanation": [
        {"heading": f"Part {i}", "body": f"section {i} " + "x" * 1200, "start_sec": i * 60}
        for i in range(60)
    ]}
    pieces = problems._windows(u, None)
    assert len(pieces) == 4
    last_start = int(pieces[-1].split("[t=")[1].split(" ")[0])
    assert last_start > 60 * 30, "the last window should be late in the video"


def test_on_screen_text_travels_with_its_part_of_the_video():
    """In a SQL video the query is on the screen and not in the speech: without
    this the extractor has the question and no answer."""
    u = {"explanation": [{"heading": "Joins", "body": "we solve one", "start_sec": 600}]}
    scenes = {"scenes": [
        {"start": 620, "ocr_text": "SELECT * FROM patients JOIN admissions USING (patient_id)"},
        {"start": 9000, "ocr_text": "SELECT something_from_much_later_in_the_video"},
    ]}
    piece = problems._windows(u, scenes)[0]
    assert "JOIN admissions" in piece
    assert "much_later" not in piece, "only the text from this stretch of video"


def test_a_video_with_no_walkthrough_yields_no_windows():
    assert problems._windows({"explanation": []}, None) == []


# --- a re-read must not cost the reader their translations ----------------
# Re-reading is a fresh model run and comes back slightly different each time.
# The first version of this silently discarded every translation on a re-read,
# which on a free tier is the reader's quota as well as their work.

def _doc(statement, solutions, original="SELECT 1;"):
    return {"problems": [{
        "n": 1, "statement": statement, "original_solution": original,
        "solutions": solutions, "notes": {"Oracle": "FETCH FIRST, not LIMIT"},
    }]}


def test_translations_survive_a_re_read():
    fresh = [problems._clean_problem({
        "title": "Second highest salary",
        "statement": "Find the second highest salary in the Employee table.",
        "solution": "SELECT 1;", "language": "MySQL",
    })]
    previous = _doc(
        "Find the SECOND highest salary in the employee table!",
        {"MySQL": "SELECT 1;", "Oracle": "SELECT 1 FROM dual;"},
    )
    problems._carry_over_translations(previous, fresh)
    assert fresh[0]["solutions"]["Oracle"] == "SELECT 1 FROM dual;"
    assert fresh[0]["notes"]["Oracle"] == "FETCH FIRST, not LIMIT"


def test_a_translation_of_different_code_is_not_carried_over():
    """The video's own solution changed between readings, so the old
    translations describe code that is no longer on the page."""
    fresh = [problems._clean_problem({
        "title": "Second highest salary",
        "statement": "Find the second highest salary in the Employee table.",
        "solution": "SELECT 2;", "language": "MySQL",
    })]
    previous = _doc(
        "Find the second highest salary in the Employee table.",
        {"Oracle": "SELECT 1 FROM dual;"},
        original="SELECT 1;",
    )
    problems._carry_over_translations(previous, fresh)
    assert "Oracle" not in fresh[0]["solutions"]


def test_a_problem_the_re_read_did_not_find_is_simply_absent():
    fresh = [problems._clean_problem({
        "title": "Something else",
        "statement": "Count the rows in the patients table.",
        "solution": "SELECT 1;", "language": "MySQL",
    })]
    problems._carry_over_translations(_doc("An entirely different question here.",
                                           {"Oracle": "x"}), fresh)
    assert list(fresh[0]["solutions"]) == ["MySQL"]


def test_no_previous_document_is_not_an_error():
    fresh = [problems._clean_problem({
        "title": "a", "statement": "Count the rows in the patients table.",
        "solution": "SELECT 1;", "language": "MySQL"})]
    problems._carry_over_translations(None, fresh)
    assert list(fresh[0]["solutions"]) == ["MySQL"]


def test_the_same_question_is_recognised_across_readings():
    """Matching on words, not on the exact string, because the model rewords
    the statement slightly every time it reads the video."""
    assert problems._key("Find the SECOND highest salary!") == problems._key(
        "find the second highest salary")
