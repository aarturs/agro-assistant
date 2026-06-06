import os, time
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

print("Loading embedding model...")
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# ── Hardcoded data parsed from the Excel file ─────────────────────────────────
# Sheet 1: Fungicide products with doses and costs
FUNGICIDES = [
    {"name": "Balaya",          "active": "Mefentriflukonazols 100 g/l + Piraklostrobins 100 g/l",  "dose": "0.5 L/ha", "dose_range": "0.5-1.5 L/ha", "eur_ha": 18.66,  "timing": "T1/T2", "notes": ""},
    {"name": "Daxur",           "active": "Metil-krezoksims 150 g/l + Mefentriflukonazols 100 g/l", "dose": "0.75 L/ha","dose_range": "0.6-1 L/ha",   "eur_ha": 24.75,  "timing": "T1/T2", "notes": "Soratel japieliek dzeltenplankumainibai"},
    {"name": "Mirador",         "active": "Azoksistrobins 250 g/l",                                  "dose": "0.5 L/ha", "dose_range": "",              "eur_ha": 6.05,   "timing": "T1/T2", "notes": ""},
    {"name": "Globaztar 250 SC","active": "Azoksistrobins 250 g/l",                                  "dose": "0.5 L/ha", "dose_range": "0.5-1 L/ha",   "eur_ha": 4.76,   "timing": "T1/T2", "notes": "Registrets tikai graudaugos"},
    {"name": "Comet Pro",       "active": "Piraklostrobins 200 g/l",                                 "dose": "0.4 L/ha", "dose_range": "0.4 L/ha",     "eur_ha": 24.15,  "timing": "T1/T2", "notes": ""},
    {"name": "Curbatur",        "active": "Protiokonazols 250 g/l",                                  "dose": "",         "dose_range": "",              "eur_ha": None,   "timing": "",      "notes": ""},
    {"name": "Priaxor",         "active": "Piraklostrobins 150 g/l + Fluksapiroksads 75 g/l",        "dose": "0.5 L/ha", "dose_range": "0.75-1.5 L/ha","eur_ha": 16.97,  "timing": "T1/T2", "notes": ""},
    {"name": "Soratel",         "active": "Protiokonazols 250 g/l",                                  "dose": "0.4 L/ha", "dose_range": "0.6-0.8 L/ha", "eur_ha": 4.56,   "timing": "T1/T2", "notes": ""},
    {"name": "Ascra Xpro",      "active": "Biksafens 65 g/l + Protiokonazols 130 g/l + Fluorpirams 65 g/l", "dose": "0.8 L/ha", "dose_range": "0.6-1.5 L/ha", "eur_ha": 34.70, "timing": "T2", "notes": ""},
    {"name": "Elatus Era",      "active": "Benzovindiflupirs 75 g/l + Protiokonazols 150 g/l",       "dose": "0.75 L/ha","dose_range": "0.75-1 L/ha",  "eur_ha": 34.01,  "timing": "T2",    "notes": ""},
    {"name": "Revystar XL",     "active": "Fluksapiroksads 50 g/l + Mefentriflukonazols 100 g/l",    "dose": "0.4 L/ha", "dose_range": "0.4-0.5 L/ha", "eur_ha": 32.87,  "timing": "T2",    "notes": "Lietojot atseviski deva ir 0.75-1.5 L/ha"},
    {"name": "Revytrex",        "active": "Mefentriflukonazols 66.7 g/l + Fluksapiroksads 66.7 g/l","dose": "1 L/ha",   "dose_range": "0.75-1.5 L/ha","eur_ha": 37.65,  "timing": "T2",    "notes": ""},
    {"name": "Delaro Forte",    "active": "Informacija no efektivitates tabulas",                     "dose": "",         "dose_range": "",              "eur_ha": None,   "timing": "",      "notes": ""},
    {"name": "Input",           "active": "Informacija no efektivitates tabulas",                     "dose": "",         "dose_range": "",              "eur_ha": None,   "timing": "",      "notes": ""},
    {"name": "Input Triple",    "active": "Informacija no efektivitates tabulas",                     "dose": "",         "dose_range": "",              "eur_ha": None,   "timing": "",      "notes": ""},
    {"name": "Verben",          "active": "Informacija no efektivitates tabulas",                     "dose": "",         "dose_range": "",              "eur_ha": None,   "timing": "",      "notes": ""},
    {"name": "Capalo Revy",     "active": "Informacija no efektivitates tabulas",                     "dose": "",         "dose_range": "",              "eur_ha": None,   "timing": "",      "notes": ""},
]

# Sheet 2: Effectiveness matrix (X=vaja, XX=labi, XXX=izcili)
EFFECTIVENESS = {
    "Delaro Forte":     {"Dzeltenplankumainiba":"XXX","Pelekplankumainiba":"XXX","Miltrasa":"XXX","Bruna rusa":"XXX","Dzelten rusa":"XXX","Pleksnu plankumainiba":"XXX","Gredzenplankumainiba":"XXX","Tiklplankumainiba":"XXX","Pundurrusa":"XXX","Fuzarioze":""},
    "Input":            {"Dzeltenplankumainiba":"XX", "Pelekplankumainiba":"XXX","Miltrasa":"X",  "Bruna rusa":"XXX","Dzelten rusa":"",   "Pleksnu plankumainiba":"",   "Gredzenplankumainiba":"XXX","Tiklplankumainiba":"XX", "Pundurrusa":"XXX","Fuzarioze":""},
    "Input Triple":     {"Dzeltenplankumainiba":"XX", "Pelekplankumainiba":"XXX","Miltrasa":"XXX","Bruna rusa":"XXX","Dzelten rusa":"",   "Pleksnu plankumainiba":"",   "Gredzenplankumainiba":"XXX","Tiklplankumainiba":"XX", "Pundurrusa":"XXX","Fuzarioze":""},
    "Balaya":           {"Dzeltenplankumainiba":"XX", "Pelekplankumainiba":"XXX","Miltrasa":"XX", "Bruna rusa":"XXX","Dzelten rusa":"XXX","Pleksnu plankumainiba":"",   "Gredzenplankumainiba":"XXX","Tiklplankumainiba":"XXX","Pundurrusa":"XXX","Fuzarioze":""},
    "Daxur":            {"Dzeltenplankumainiba":"XX", "Pelekplankumainiba":"XXX","Miltrasa":"XX", "Bruna rusa":"XXX","Dzelten rusa":"XXX","Pleksnu plankumainiba":"",   "Gredzenplankumainiba":"XXX","Tiklplankumainiba":"XXX","Pundurrusa":"XXX","Fuzarioze":""},
    "Mirador":          {"Dzeltenplankumainiba":"",   "Pelekplankumainiba":"",   "Miltrasa":"",   "Bruna rusa":"XXX","Dzelten rusa":"XXX","Pleksnu plankumainiba":"XXX","Gredzenplankumainiba":"XX", "Tiklplankumainiba":"XXX","Pundurrusa":"XXX","Fuzarioze":""},
    "Globaztar 250 SC": {"Dzeltenplankumainiba":"",   "Pelekplankumainiba":"",   "Miltrasa":"",   "Bruna rusa":"XXX","Dzelten rusa":"XXX","Pleksnu plankumainiba":"XXX","Gredzenplankumainiba":"XX", "Tiklplankumainiba":"XXX","Pundurrusa":"XXX","Fuzarioze":""},
    "Comet Pro":        {"Dzeltenplankumainiba":"XX", "Pelekplankumainiba":"XXX","Miltrasa":"",   "Bruna rusa":"XXX","Dzelten rusa":"XXX","Pleksnu plankumainiba":"XXX","Gredzenplankumainiba":"",   "Tiklplankumainiba":"XXX","Pundurrusa":"XXX","Fuzarioze":""},
    "Curbatur":         {"Dzeltenplankumainiba":"XX", "Pelekplankumainiba":"XXX","Miltrasa":"XXX","Bruna rusa":"XXX","Dzelten rusa":"XXX","Pleksnu plankumainiba":"XXX","Gredzenplankumainiba":"XXX","Tiklplankumainiba":"XXX","Pundurrusa":"XXX","Fuzarioze":"XX"},
    "Verben":           {"Dzeltenplankumainiba":"XXX","Pelekplankumainiba":"XX", "Miltrasa":"XXX","Bruna rusa":"XXX","Dzelten rusa":"XXX","Pleksnu plankumainiba":"",   "Gredzenplankumainiba":"XXX","Tiklplankumainiba":"XXX","Pundurrusa":"XXX","Fuzarioze":""},
    "Capalo Revy":      {"Dzeltenplankumainiba":"XX", "Pelekplankumainiba":"XXX","Miltrasa":"XXX","Bruna rusa":"XXX","Dzelten rusa":"XXX","Pleksnu plankumainiba":"",   "Gredzenplankumainiba":"XX", "Tiklplankumainiba":"XXX","Pundurrusa":"XXX","Fuzarioze":""},
    "Soratel":          {"Dzeltenplankumainiba":"XXX","Pelekplankumainiba":"XX", "Miltrasa":"XX", "Bruna rusa":"XXX","Dzelten rusa":"XXX","Pleksnu plankumainiba":"",   "Gredzenplankumainiba":"XXX","Tiklplankumainiba":"XXX","Pundurrusa":"XXX","Fuzarioze":"XX"},
    "Ascra Xpro":       {"Dzeltenplankumainiba":"XXX","Pelekplankumainiba":"XXX","Miltrasa":"XXX","Bruna rusa":"XXX","Dzelten rusa":"XXX","Pleksnu plankumainiba":"XXX","Gredzenplankumainiba":"XXX","Tiklplankumainiba":"XXX","Pundurrusa":"XXX","Fuzarioze":"XX"},
    "Elatus Era":       {"Dzeltenplankumainiba":"XX", "Pelekplankumainiba":"XXX","Miltrasa":"",   "Bruna rusa":"XXX","Dzelten rusa":"XXX","Pleksnu plankumainiba":"XX", "Gredzenplankumainiba":"XXX","Tiklplankumainiba":"XXX","Pundurrusa":"XXX","Fuzarioze":"XX"},
    "Revystar XL":      {"Dzeltenplankumainiba":"XX", "Pelekplankumainiba":"XXX","Miltrasa":"XXX","Bruna rusa":"XXX","Dzelten rusa":"XXX","Pleksnu plankumainiba":"",   "Gredzenplankumainiba":"XXX","Tiklplankumainiba":"XXX","Pundurrusa":"XXX","Fuzarioze":""},
    "Priaxor":          {"Dzeltenplankumainiba":"XXX","Pelekplankumainiba":"XXX","Miltrasa":"XXX","Bruna rusa":"XXX","Dzelten rusa":"XXX","Pleksnu plankumainiba":"XXX","Gredzenplankumainiba":"XXX","Tiklplankumainiba":"XXX","Pundurrusa":"XXX","Fuzarioze":""},
    "Revytrex":         {"Dzeltenplankumainiba":"XX", "Pelekplankumainiba":"XXX","Miltrasa":"XX", "Bruna rusa":"XXX","Dzelten rusa":"XXX","Pleksnu plankumainiba":"",   "Gredzenplankumainiba":"XXX","Tiklplankumainiba":"",   "Pundurrusa":"XXX","Fuzarioze":""},
}

RATING = {"XXX": "izcili efektivs", "XX": "labi efektivs", "X": "vaji efektivs", "": "nav datu"}


def build_chunk(f):
    name = f["name"]
    eff = EFFECTIVENESS.get(name, {})

    # Build effectiveness summary
    top = [disease for disease, rating in eff.items() if rating == "XXX"]
    good = [disease for disease, rating in eff.items() if rating == "XX"]

    eff_text = ""
    if top:
        eff_text += "Izcili efektivs pret: " + ", ".join(top) + ". "
    if good:
        eff_text += "Labi efektivs pret: " + ", ".join(good) + ". "

    cost_text = f"Izmaksas: {f['eur_ha']} EUR/ha. " if f["eur_ha"] else ""
    dose_text = f"Deva: {f['dose']}. " if f["dose"] else ""
    range_text = f"Deva diapazons: {f['dose_range']}. " if f["dose_range"] else ""
    timing_text = f"Lietosanas laiks: {f['timing']}. " if f["timing"] else ""
    notes_text = f"Piezimes: {f['notes']}. " if f["notes"] else ""

    return (
        f"Fungicids (graudaugi): {name}. "
        f"Darbiga viela: {f['active']}. "
        + dose_text
        + range_text
        + cost_text
        + timing_text
        + eff_text
        + notes_text
    ).strip()


def ingest():
    print(f"Processing {len(FUNGICIDES)} fungicide products from agronomist spreadsheet...")
    batch, errors, count = [], 0, 0

    for i, f in enumerate(FUNGICIDES):
        try:
            chunk = build_chunk(f)
            print(f"  [{i+1}] {f['name']}: {chunk[:80]}...")

            embedding = model.encode(
                "search_document: " + chunk, normalize_embeddings=True
            ).tolist()

            batch.append({
                "product_name":   f["name"][:200],
                "active_subst":   f["active"][:200],
                "culture":        "Graudaugi (kviei, miei, rudzi, auzas)",
                "pest_disease":   ", ".join([d for d, r in EFFECTIVENESS.get(f["name"], {}).items() if r]),
                "dose":           f["dose"],
                "quarantine_days": None,
                "reg_number":     "AGR_" + f["name"].replace(" ", "_").upper(),
                "content":        chunk,
                "embedding":      embedding,
            })
            count += 1

        except Exception as e:
            errors += 1
            print(f"  Error {f['name']}: {e}")

    if batch:
        supabase.table("vaad_products").upsert(
            batch, on_conflict="reg_number"
        ).execute()
        print(f"\nUpserted {count} fungicide records.")

    print(f"Done! {count} products ingested, {errors} errors.")
    print("\nThese products are now searchable in the AI assistant.")
    print("Try asking: 'Kadu fungicidu lietot kviešiem pret brunu rusu?'")


if __name__ == "__main__":
    ingest()
