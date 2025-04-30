# Prompt-stratégiák hatása a CodeLlama problémamegoldó teljesítményére

Beadandó az ELTE Informatikaon Kar Adattárházak és Adatbányászat MSC tárgyára

Kapcsolódó projectek:

- [Overleaf projekt](https://www.overleaf.com/project/680ff38883f9adae36741b1e)
- [Slides](https://ikelte-my.sharepoint.com/:p:/r/personal/ekzdy0_inf_elte_hu/_layouts/15/doc.aspx?sourcedoc=%7B50b68c70-7b72-4493-9312-1b6ec7ff8da1%7D&action=edit)
- [Adathalmaz](https://www.kaggle.com/datasets/osztopanikristof/project-euler-problems-with-solution/data)

## Projekt célja
Ez a projekt azt vizsgálja, hogy különböző prompt-stratégiák hogyan befolyásolják a CodeLlama és más nagy nyelvi modellek (LLM-ek) matematikai problémamegoldó képességét. A vizsgálat alapját a Project Euler feladatok adják.

## Főbb funkciók
- Project Euler feladatok letöltése, feldolgozása és egységesítése
- Különböző prompt-stratégiák (pl. simple, chain-of-thought, role, few-shot stb.) automatikus generálása
- LLM-ek kiértékelése Python kódgenerálási és problémamegoldó feladatokon
- Eredmények automatikus összegyűjtése, kiértékelése és táblázatos összefoglalása

## Mappastruktúra
- `src/` – Forráskód (adatfeldolgozás, kiértékelés, promptok, főprogram)
- `data/dataset/` – Feldolgozott Project Euler feladatok (pl. `euler_problems.json`)
- `data/measurements/` – Kísérleti eredmények, összefoglaló táblázatok (CSV)
- `README.md` – Ez a dokumentáció
- `LICENSE` – Licenc (MIT)

## Használat
1. **Függőségek telepítése**
   - Python 3.9+
   - Ajánlott: virtuális környezet használata
   - Kötelező csomagok: `pandas`, `requests`, `prettytable`, `python-dotenv`, stb.
   - Telepítés:  
     ```bash
     pip install -r requirements.txt
     ```
2. **Adatfeldolgozás**
   - A Project Euler feladatok letöltése és egységesítése:  
     ```bash
     python src/process.py
     ```
3. **Értékelés és kísérletek**
   - Prompt-stratégiák kiértékelése, eredmények mentése:  
     ```bash
     python src/test.py
     ```
   - Eredmények elemzése:  
     A `data/measurements/` mappában található CSV fájlokban.

## Adatforrások
- [Project Euler](https://projecteuler.net/)
- Prompt- és eredményfájlok: lásd `data/` mappa

## Eredmények
A kész eredmények összefoglaló táblázatokban találhatók (pl. `./data/mesurements/*_results_summary.csv`). Ezek tartalmazzák, hogy az egyes prompt-stratégiák hány feladatot oldottak meg helyesen különböző nehézségi szinteken.

## Szerző
Osztopáni Kristóf
Horcsin Bálint

## Licenc
MIT License
