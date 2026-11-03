"""Static regression checks for the AIS3 Vietnamese roadmap, nav, KPI evidence and assets.

No model, real device, browser, external network or additional Python dependency required.
"""
import csv
from html.parser import HTMLParser
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.href_ids = []
        self.option_ids = []
        self.copy_ids = []
        self.routes = []
        self.phases = {}
        self.local_assets = []

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        id_ = d.get("id")
        if id_:
            self.ids.append(id_)
        if d.get("href", "").startswith("#"):
            self.href_ids.append(d["href"][1:])
        if tag == "option" and d.get("value", "").startswith("#"):
            self.option_ids.append(d["value"][1:])
        if d.get("data-copy"):
            self.copy_ids.append(d["data-copy"])
        if d.get("data-route-filter"):
            self.routes.append(d["data-route-filter"])
        if "roadmap-phase" in d.get("class", "").split():
            self.phases[id_] = d.get("data-tracks", "")
        attr = "src" if tag == "script" else "href" if tag == "link" else None
        if attr and d.get(attr, "").startswith("./"):
            self.local_assets.append(d[attr])


class RoadmapSiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (DOCS / "index.html").read_text(encoding="utf-8")
        cls.page = PageParser()
        cls.page.feed(cls.html)

    def test_every_internal_link_and_copy_button_resolves(self):
        ids = set(self.page.ids)
        self.assertEqual(len(ids), len(self.page.ids), "Duplicate HTML id")
        for target in self.page.href_ids + self.page.option_ids + self.page.copy_ids:
            self.assertIn(target, ids, f"Missing anchor or copy id: {target}")

    def test_nine_roadmap_phases_and_routes(self):
        self.assertIn("roadmap", self.page.ids)
        phases = self.page.phases
        self.assertEqual(set(phases), {f"roadmap-r{i}" for i in range(1, 10)})
        self.assertEqual(set(self.page.routes),
                         {"all", "ais3", "supermini", "vi", "kws", "xiaozhi"})
        for route in set(self.page.routes) - {"all"}:
            count = sum("common" in tags.split() or route in tags.split()
                        for tags in phases.values())
            self.assertGreater(count, 0, f"No phases selectable for {route}")
        self.assertIn('href="#roadmap"', self.html)
        self.assertIn('value="#roadmap"', self.html)

    def test_roadmap_evidence_controls_are_not_autopassed(self):
        self.assertEqual(self.html.count('data-roadmap-evidence'), 5)
        self.assertIn('id="roadmap-evidence-bar"', self.html)
        self.assertIn('id="roadmap-selected-time"', self.html)
        self.assertIn("chưa xác nhận", self.html.lower())
        self.assertNotIn('data-roadmap-evidence checked', self.html)

    def test_local_assets_and_docs_exist(self):
        for relative in self.page.local_assets:
            self.assertTrue((DOCS / relative[2:]).is_file(), f"Missing file: {relative}")
        for filename in ["ROADMAP_VI.md", "roadmap-kpi-template.csv", "roadmap.css",
                         "roadmap.js", "VI_KWS.md"]:
            self.assertTrue((DOCS / filename).is_file(), filename)

    def test_kpi_csv_is_a_blank_evidence_template(self):
        with (DOCS / "roadmap-kpi-template.csv").open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertGreaterEqual(len(rows), 15)
        self.assertEqual(set(row["phase"] for row in rows),
                         {f"R{i}" for i in range(1, 10)})
        expected_columns = {"phase", "track", "metric", "target", "unit", "measured",
                            "evidence_path", "date", "owner", "reviewer", "decision", "notes"}
        self.assertEqual(set(rows[0]), expected_columns)
        for row in rows:
            self.assertEqual(row["decision"], "PENDING")
            self.assertEqual(row["measured"], "")
            self.assertEqual(row["evidence_path"], "")
            self.assertTrue(row["target"])

    def test_training_and_kws_claims_distinguish_plan_from_result(self):
        self.assertIn("8–72 giờ", self.html)
        self.assertIn("Macro-F1", self.html)
        self.assertIn("94%", self.html)
        self.assertIn("không phải benchmark ais3", self.html.lower())
        self.assertIn("20/20", self.html)
        doc = (DOCS / "ROADMAP_VI.md").read_text(encoding="utf-8")
        for key in ["G0", "G1", "G2", "G3", "G4", "PENDING", "NFC", "XiaoZhi"]:
            self.assertIn(key, doc)


if __name__ == "__main__":
    unittest.main()
