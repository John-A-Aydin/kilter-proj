"""Small regression fixtures for the source format's easily confused semantics."""

from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch, Mock

from download import download_snapshot
from prepare import QUERY, angle_metadata, convert_frame, decode_frames


class FrameTests(unittest.TestCase):
    def test_roles_and_last_update_wins(self):
        self.assertEqual(decode_frames("p1r12p2r13p3r15p4r14p2r15"),
                         [{1: 12, 2: 15, 3: 15, 4: 14}])

    def test_absolute_frame_replaces_previous_holds(self):
        self.assertEqual(decode_frames('p1r12p2r13,p3r14'), [{1: 12, 2: 13}, {3: 14}])

    def test_delta_frame_carries_holds_and_applies_removals(self):
        self.assertEqual(decode_frames('p1r12p2r13,"x2p3r14'),
                         [{1: 12, 2: 13}, {1: 12, 3: 14}])

    def test_quoted_empty_frame_repeats_previous_state(self):
        self.assertEqual(decode_frames('p1r12,",,"p1r13,'), [{1: 12}, {1: 12}, {1: 13}])

    def test_event_order_matters(self):
        self.assertEqual(decode_frames('p1r12x1p2r13'), [{2: 13}])

    def test_rejects_unparsed_input(self):
        for frames in ["", "p1r12garbage", "q1p2r13", "p1r"]:
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                decode_frames(frames)

    def test_snapshots_are_independent(self):
        frames = decode_frames('p1r12,"p1r13')
        self.assertEqual(frames[0], {1: 12})

    def test_unknown_roles_and_outside_holds_fail(self):
        holds = {1: {"hold_x": 0.25, "hold_y": 0.5, "on_kickboard": 0, "hold_token": 1}}
        for frame in [{1: 999}, {2: 12}]:
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                convert_frame(frame, {12: 2}, holds)

    def test_physical_token_is_independent_of_use(self):
        holds = {1: {"hold_x": 0.25, "hold_y": 0.5, "on_kickboard": 0, "hold_token": 1}}
        start = convert_frame({1: 12}, {12: 2, 14: 3}, holds)[0]
        finish = convert_frame({1: 14}, {12: 2, 14: 3}, holds)[0]
        self.assertEqual(start["hold_token"], finish["hold_token"])
        self.assertEqual((start["hold_use_type"], finish["hold_use_type"]), (2, 3))


class DownloadTests(unittest.TestCase):
    def test_rejects_wrong_range_instead_of_publishing(self):
        response = Mock(status_code=206, headers={"Content-Range": "bytes 1-3/4"})
        response.raw.read.return_value = b"abcd"
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with tempfile.TemporaryDirectory() as folder, patch("download.requests.get", return_value=response):
            destination = Path(folder) / "snapshot.sqlite3"
            with self.assertRaisesRegex(ValueError, "invalid byte range"):
                download_snapshot({"url": "https://example.invalid/snapshot", "bytes": 4,
                                   "contentEncoding": "identity"}, destination, workers=1)
            self.assertFalse(destination.exists())


class SelectionTests(unittest.TestCase):
    def test_coverage_and_ascent_evidence_filter(self):
        with sqlite3.connect(":memory:") as db:
            db.execute("ATTACH DATABASE ':memory:' AS geometry")
            db.execute("ATTACH DATABASE ':memory:' AS grades")
            db.execute("""CREATE TABLE grades.board_climb_grades (climb_uuid TEXT, board_type TEXT,
                angle INTEGER, local_grade REAL, universal_grade REAL, confidence TEXT)""")
            db.execute("""CREATE TABLE board_climbs (uuid TEXT, board_type TEXT, layout_id INTEGER,
                is_listed INTEGER, is_draft INTEGER, is_hidden INTEGER, compatible_size_ids TEXT)""")
            db.execute("""CREATE TABLE board_climb_stats (climb_uuid TEXT, board_type TEXT, angle INTEGER,
                ascensionist_count INTEGER, display_difficulty REAL, benchmark_difficulty REAL, quality_average REAL,
                difficulty_average REAL, fa_username TEXT, fa_at TEXT)""")
            db.execute("""CREATE TABLE geometry.climb_stats (climb_uuid TEXT, angle INTEGER, ascensionist_count INTEGER,
                display_difficulty REAL, benchmark_difficulty REAL, quality_average REAL,
                difficulty_average REAL, fa_username TEXT, fa_at TEXT)""")
            variants = [
                ("current", "kilter", 1, 1, 0, 0, "[10]", 45, 1),
                ("historical", "kilter", 1, 1, 0, 0, "[10]", 45, 0),
                ("missing_current", "kilter", 1, 1, 0, 0, "[10]", 45, None),
                ("draft", "kilter", 1, 1, 1, 0, "[10]", 45, 1),
                ("hidden", "kilter", 1, 1, 0, 1, "[10]", 45, 1),
                ("unlisted", "kilter", 1, 0, 0, 0, "[10]", 45, 1),
                ("other_size", "kilter", 1, 1, 0, 0, "[7]", 45, 1),
                ("other_layout", "kilter", 8, 1, 0, 0, "[10]", 45, 1),
                ("another_angle", "kilter", 1, 1, 0, 0, "[10]", 40, 1),
                ("no_ascents", "kilter", 1, 1, 0, 0, "[10]", 45, 0),
            ]
            for *climb, angle, ascents in variants:
                db.execute("INSERT INTO board_climbs VALUES (?, ?, ?, ?, ?, ?, ?)", climb)
                if ascents is not None:
                    db.execute("INSERT INTO board_climb_stats VALUES (?, ?, ?, ?, 1, NULL, 3, 1, 'setter', 'date')",
                               (climb[0], climb[1], angle, ascents))
            db.executemany("INSERT INTO geometry.climb_stats VALUES (?, 45, 2, 2, NULL, 3, 2, 'setter', 'date')",
                           [("historical",), ("missing_current",)])
            # One climb has overlapping evidence at 45, a current-only 0-degree
            # record, and a legacy-only 70-degree record. Do not cross-join angles.
            db.execute("INSERT INTO board_climb_stats VALUES ('current','kilter',0,3,4,NULL,3,4,'s','d')")
            db.execute("INSERT INTO geometry.climb_stats VALUES ('current',45,8,2,NULL,3,2,'s','d')")
            db.execute("INSERT INTO geometry.climb_stats VALUES ('current',70,9,5,NULL,3,5,'s','d')")
            db.row_factory = sqlite3.Row
            result = db.execute(QUERY, {"layout_id": 1, "size_id": 10}).fetchall()
            self.assertEqual([(r['uuid'], r['stats_angle']) for r in result],
                             [("another_angle",40), ("current",0), ("current",45), ("current",70),
                              ("historical",45), ("missing_current",45)])
            self.assertEqual([(r['current_ascensionist_count'], r['legacy_ascensionist_count'])
                              for r in result if r['uuid']=='current'], [(3,None), (1,8), (None,9)])


class GradeTests(unittest.TestCase):
    def row(self):
        return {
            "uuid": "climb", "stats_angle": 40,
            "current_ascensionist_count": 3, "legacy_ascensionist_count": 8,
            "computed_local_grade": 18.0, "computed_universal_grade": 17.0,
            "computed_grade_confidence": "provisional",
            **{f"{prefix}_{key}": value for prefix, values in [
                ("current", [16.6, 16.6, None, 4.2, "new_fa", "new_date"]),
                ("legacy", [15.0, 15.0, None, 3.0, "old_fa", "old_date"]),
            ] for key, value in zip(["display_difficulty", "difficulty_average", "benchmark_difficulty",
                                    "quality_average", "fa_username", "fa_at"], values)},
        }

    def test_current_grade_count_and_angle_stay_together(self):
        metadata = angle_metadata(self.row(), {15:"6b/V4",17:"6c/V5"})
        self.assertEqual((metadata['climb_angle_id'],metadata['angle'],metadata['grade'],
                          metadata['ascensionist_count'], metadata['quality_average']),
                         ('climb:40',40,'6c/V5',3,4.2))
        self.assertEqual((metadata['grade_font'], metadata['grade_hueco']), ('6c','V5'))

    def test_historical_grade_comes_from_same_source_as_count(self):
        row = self.row(); row['current_ascensionist_count']=0
        metadata = angle_metadata(row, {15:'6b/V4',17:'6c/V5'})
        self.assertEqual((metadata['grade'],metadata['ascensionist_count'],metadata['grade_source'],
                          metadata['display_difficulty'],metadata['fa_username']),
                         ('6b/V4',8,'boardlib_snapshot',15.0,'old_fa'))

    def test_missing_display_grade_uses_same_angle_average(self):
        row = self.row(); row['current_display_difficulty']=None
        self.assertEqual(angle_metadata(row,{17:'6c/V5'})['grade'],'6c/V5')

    def test_computed_grade_is_flagged_when_native_grade_missing(self):
        row = self.row()
        for key in ['current_display_difficulty','current_benchmark_difficulty',
                    'current_difficulty_average','legacy_display_difficulty']:
            row[key]=None
        metadata = angle_metadata(row,{18:'6c+/V5'})
        self.assertEqual((metadata['grade'],metadata['grade_source'],metadata['grade_status'],
                          metadata['computed_grade_confidence']),
                         ('6c+/V5','boardsesh_computed','computed','provisional'))

    def test_missing_grade_is_explicit_without_excluding_the_row(self):
        row = self.row()
        for key in ['current_display_difficulty','current_benchmark_difficulty',
                    'current_difficulty_average','legacy_display_difficulty','computed_local_grade']:
            row[key]=None
        metadata = angle_metadata(row,{18:'6c+/V5'})
        self.assertIsNone(metadata['grade'])
        self.assertEqual(metadata['grade_status'],'unavailable')
        self.assertIsNone(metadata['grade_source'])
        self.assertEqual((metadata['angle'],metadata['ascensionist_count']),(40,3))


if __name__ == "__main__":
    unittest.main()
