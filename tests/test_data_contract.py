import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "df_cleaned.csv"
GEOJSON_PATH = ROOT / "data" / "prov 34.geojson"

REQUIRED_DATA_COLUMNS = [
    "Provinsi",
    "Kab/Kota",
    "Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)",
    "Rata-rata Lama Sekolah Penduduk 15+ (Tahun)",
    "Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)",
    "Indeks Pembangunan Manusia",
    "Umur Harapan Hidup (Tahun)",
    "Persentase rumah tangga yang memiliki akses terhadap sanitasi layak",
    "Persentase rumah tangga yang memiliki akses terhadap air minum layak",
    "Tingkat Pengangguran Terbuka",
    "Tingkat Partisipasi Angkatan Kerja",
    "PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)",
]

NUMERIC_DATA_COLUMNS = [col for col in REQUIRED_DATA_COLUMNS if col not in {"Provinsi", "Kab/Kota"}]


def read_clean_rows():
    with DATA_PATH.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_geojson():
    with GEOJSON_PATH.open(encoding="utf-8") as file:
        return json.load(file)


class DataContractTest(unittest.TestCase):
    def test_clean_dataset_has_required_columns_and_values(self):
        rows = read_clean_rows()

        self.assertGreater(len(rows), 0)
        self.assertEqual(set(REQUIRED_DATA_COLUMNS) - set(rows[0]), set())

        for row_number, row in enumerate(rows, start=2):
            for column in REQUIRED_DATA_COLUMNS:
                self.assertNotEqual(row[column].strip(), "", f"{column} kosong di baris {row_number}")

            for column in NUMERIC_DATA_COLUMNS:
                with self.subTest(row=row_number, column=column):
                    float(row[column])

    def test_clean_dataset_contains_34_provinces(self):
        rows = read_clean_rows()
        provinces = {row["Provinsi"].strip().upper() for row in rows}

        self.assertEqual(len(provinces), 34)

    def test_geojson_contains_34_named_provinces(self):
        geojson_data = read_geojson()
        features = geojson_data.get("features", [])
        provinces = {
            feature.get("properties", {}).get("name", "").strip().upper()
            for feature in features
        }

        self.assertEqual(geojson_data.get("type"), "FeatureCollection")
        self.assertEqual(len(features), 34)
        self.assertEqual(len(provinces), 34)
        self.assertNotIn("", provinces)

    def test_dataset_and_geojson_provinces_match(self):
        data_provinces = {row["Provinsi"].strip().upper() for row in read_clean_rows()}
        geojson_provinces = {
            feature.get("properties", {}).get("name", "").strip().upper()
            for feature in read_geojson().get("features", [])
        }

        self.assertEqual(data_provinces, geojson_provinces)


if __name__ == "__main__":
    unittest.main()
