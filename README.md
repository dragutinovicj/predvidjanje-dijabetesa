# Predviđanje dijabetesa pomoću neuronske mreže

## Opis problema

Dijabetes mellitus predstavlja jedan od najrasprostranjenijih hroničnih zdravstvenih problema savremenog doba. Rano otkrivanje bolesti ključno je za pravovremeno lečenje i sprečavanje komplikacija. Cilj ovog rada je izgradnja sistema za automatsko predviđanje prisustva dijabetesa na osnovu rutinskih medicinskih merenja primenom višeslojne neuronske mreže, uz poseban osvrt na interpretabilnost modela i medicinsku relevantnost rezultata.

---

## Podaci

### Izvor
Pima Indians Diabetes Dataset, dostupan na Kaggle platformi:
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### Struktura
- 768 ispitanica ženskog pola pima indijanskog porekla starijih od 21 godine
- 8 ulaznih promenljivih i 1 ciljna promenljiva

| Promenljiva | Opis |
|---|---|
| Pregnancies | Broj trudnoća |
| Glucose | Koncentracija glukoze u plazmi |
| BloodPressure | Dijastolni krvni pritisak (mm Hg) |
| SkinThickness | Debljina kožnog nabora tricepsa (mm) |
| Insulin | Nivo insulina u serumu (mu U/ml) |
| BMI | Indeks telesne mase |
| DiabetesPedigreeFunction | Funkcija porodičnog rizika od dijabetesa |
| Age | Starost (godine) |
| Outcome | Ciljna promenljiva (0 = nema dijabetesa, 1 = dijabetes) |

- Raspodela klasa: ~65% negativnih, ~35% pozitivnih

### Preprocesiranje
- Vrednosti 0 zamenjene sa NaN kod fiziološki nemogućih promenljivih (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
- Stratifikovana podela na trening (60%), validacioni (20%) i test skup (20%)
- Imputacija nedostajućih vrednosti primenom MICE algoritma (IterativeImputer sa BayesianRidge estimatorom)
- Normalizacija primenom StandardScaler-a
- Sve transformacije parametrizovane isključivo na trening skupu kako bi se sprečilo curenje informacija (data leakage)

---

## Arhitektura modela

Višeslojna neuronska mreža (MLP) implementirana u PyTorch-u sa hiperparametrima određenim Optuna optimizacijom:
Linear(8 → 64) → ReLU → Dropout(0.316)
Linear(64 → 32) → ReLU → Dropout(0.226)
Linear(32 → 1)  → Sigmoid

| Hiperparametar | Vrednost |
|---|---|
| Learning rate | 0.00995 |
| Hidden1 | 64 neurona |
| Hidden2 | 32 neurona |
| Dropout1 | 0.316 |
| Dropout2 | 0.226 |
| Batch size | 64 |
| Optimizer | Adam |
| Loss funkcija | BCELoss |

---

## Trening

- Maksimalno epoha: 200
- Early stopping patience: 20
- Trening zaustavljen u epohi 25
- Najbolji validacioni gubitak: 0.4009
- Hiperparametarska optimizacija: Optuna (100 trial-ova, MedianPruner)

---

## Analiza osetljivosti i hiperparametarska optimizacija

Hiperparametarska optimizacija primenom Optuna biblioteke kroz 100 pokušaja pokazala je da broj neurona u prvom skrivenom sloju ima najveći uticaj na kvalitet modela (važnost 0.22), odmah praćen dropout stopom drugog sloja (0.20) i stopom učenja (0.18).

SHAP analiza osetljivosti identifikovala je sledeći redosled važnosti promenljivih:

| Promenljiva | Prosečni SHAP doprinos |
|---|---|
| Glucose | 0.19 |
| BMI | 0.11 |
| Age | 0.06 |
| DiabetesPedigreeFunction | 0.05 |
| Pregnancies | 0.03 |
| BloodPressure | 0.02 |
| Insulin | 0.02 |
| SkinThickness | 0.01 |

---

## Rezultati evaluacije

Evaluacija sprovedena na test skupu sa optimalnim pragom odluke od 0.30, određenim na osnovu maksimalnog F1 skora:

| Metrika | Početni model | Optimizovani model |
|---|---|---|
| Accuracy | 0.7403 | 0.7727 |
| Precision | 0.6029 | — |
| Recall | 0.7593 | — |
| F1 score | 0.6721 | 0.7107 |
| AUC-ROC | 0.8081 | 0.8226 |

> Korišćen je optimalni prag odluke od 0.30 umesto standardnog 0.5 zbog neravnomerne raspodele klasa i medicinskog konteksta — propuštanje bolesnog pacijenta ozbiljniji je ishod od lažno pozitivnog rezultata.

---

## Diskusija

Optimizovani model postigao je tačnost od 77.27% i AUC-ROC vrednost od 0.8226, što je u skladu sa rezultatima iz literature. Na Pima Indians Diabetes datasetu tipično se postižu tačnosti između 74% i 83% — logistička regresija oko 75-78%, Random Forest 76-81%, a XGBoost 77-83% — što znači da implementirana neuronska mreža postiže rezultate uporedive sa algoritmima koji generalno bolje funkcionišu na malim tabelarnim skupovima. Ograničenja u performansama pre svega su posledica karakteristika samog skupa — malog broja primera, visokog udela nedostajućih vrednosti i uske demografske osnove.

---

## Zaključak

Implementiran je sistem za predviđanje dijabetesa sa tačnošću od 77.27% i AUC-ROC vrednošću od 0.8226. Rezultati su uporedivi sa literaturom, a metodološki doprinos rada ogleda se u pažljivoj pripremi podataka bez curenja informacija, automatskoj hiperparametarskoj optimizaciji i medicinski opravdanom tretmanu nebalansiranosti klasa kroz optimizaciju praga odluke. Buduća istraživanja mogla bi uključiti poređenje sa Random Forest-om i XGBoost-om te validaciju na većim i demografski raznovrsnijim skupovima podataka.

---

## Pokretanje projekta

### Preduslovi

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn optuna shap
```

### Reprodukcija rezultata

1. Kloniraj repozitorijum:
```bash
git clone https://github.com/username/predvidjanje-dijabetesa.git
cd predvidjanje-dijabetesa
```

2. Preuzmi dataset sa Kaggle-a i postavi `diabetes.csv` u isti direktorijum kao notebook

3. Otvori i pokreni notebook:
```bash
jupyter notebook Predviđanje_dijabetesa.ipynb
```

ili otvori direktno u Google Colab-u.

---

## Struktura repozitorijuma
├── Predviđanje_dijabetesa.ipynb   # Glavni notebook
├── diabetes.csv                    # Skup podataka
└── README.md                       # Opis projekta

---

## Licenca

MIT License — slobodno korišćenje, modifikacija i distribucija uz navođenje izvora.
