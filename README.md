# Predviđanje dijabetesa pomoću neuronske mreže

## 1. Opis problema

Dijabetes mellitus predstavlja jedan od najrasprostranjenijih hroničnih zdravstvenih problema savremenog doba. Rano otkrivanje bolesti ključno je za pravovremeno lečenje i sprečavanje komplikacija. Cilj ovog rada je izgradnja sistema za automatsko predviđanje prisustva dijabetesa na osnovu rutinskih medicinskih merenja primenom višeslojne neuronske mreže, uz poseban osvrt na interpretabilnost modela i medicinsku relevantnost rezultata.

---

## 2. Podaci

### Izvor
Pima Indians Diabetes Dataset, dostupan na Kaggle platformi:
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### Struktura

- 768 ispitanica ženskog pola starijih od 21 godine
- 8 ulaznih promenljivih i 1 ciljna promenljiva
- Raspodela klasa: ~65% negativnih, ~35% pozitivnih

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

### Preprocesiranje

- Vrednosti 0 zamenjene sa NaN kod fiziološki nemogućih promenljivih (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
- Stratifikovana podela na trening (60%), validacioni (20%) i test skup (20%)
- Imputacija nedostajućih vrednosti primenom MICE algoritma (IterativeImputer sa BayesianRidge estimatorom)
- Normalizacija primenom StandardScaler-a
- Sve transformacije parametrizovane isključivo na trening skupu radi sprečavanja curenja informacija (data leakage)

---

## 3. Arhitektura modela

Višeslojna neuronska mreža (MLP) implementirana u PyTorch-u:
Linear(8 → 32) → ReLU → Dropout(0.414)
Linear(32 → 16) → ReLU → Dropout(0.261)
Linear(16 → 1)  → Sigmoid

Hiperparametri određeni Optuna optimizacijom:

| Hiperparametar | Vrednost |
|---|---|
| Learning rate | 0.00993 |
| Hidden1 | 32 neurona |
| Hidden2 | 16 neurona |
| Dropout1 | 0.414 |
| Dropout2 | 0.261 |
| Batch size | 64 |
| Optimizer | Adam |
| Loss funkcija | BCELoss |

---

## 4. Trening

- Optimizer: Adam, learning rate: 0.001
- Loss funkcija: BCELoss
- Maksimalno epoha: 200
- Early stopping patience: 20
- Trening zaustavljen u epohi 56
- Najbolji validacioni gubitak: 0.4265

---

## 5. Analiza osetljivosti i hiperparametarska optimizacija

Hiperparametarska optimizacija sprovedena primenom Optuna biblioteke kroz 100 pokušaja uz TPESampler i MedianPruner. Najvažniji parametar pokazala se stopa učenja sa vrednošću važnosti 0.89.

SHAP analiza osetljivosti identifikovala je sledeći redosled važnosti promenljivih:

| Promenljiva | Prosečni SHAP doprinos |
|---|---|
| Glucose | 0.20 |
| BMI | 0.10 |
| Age | 0.05 |
| DiabetesPedigreeFunction | 0.05 |
| Insulin | 0.025 |
| Pregnancies | 0.025 |
| BloodPressure | ~0.01 |
| SkinThickness | ~0.01 |

---

## 6. Rezultati evaluacije

Evaluacija sprovedena na test skupu sa optimalnim pragom odluke od 0.31:

| Metrika | Početni model (prag=0.31) | Optimizovani model (prag=0.31) |
|---|---|---|
| Accuracy | 0.7468 | 0.7662 |
| F1 score | 0.6929 | 0.7000 |
| AUC-ROC | 0.8093 | 0.8226 |

> Korišćen je optimalni prag odluke od 0.31 umesto standardnog 0.5 zbog neravnomerne raspodele klasa i medicinskog konteksta — propuštanje bolesnog pacijenta ozbiljniji je ishod od lažno pozitivnog rezultata.

---

## 7. Diskusija

Optimizovani model postigao je tačnost od 76.62% i AUC-ROC vrednost od 0.8226, što je u skladu sa rezultatima iz literature. Na Pima Indians Diabetes datasetu tipično se postižu tačnosti između 74% i 83% — logistička regresija oko 75-78%, a Random Forest 76-81% — što znači da implementirana neuronska mreža postiže rezultate uporedive sa algoritmima koji generalno bolje funkcionišu na malim tabelarnim skupovima. Ograničenja u performansama pre svega su posledica karakteristika samog skupa — malog broja primera, visokog udela nedostajućih vrednosti i uske demografske osnove.

---

## 8. Zaključak

Implementiran je sistem za predviđanje dijabetesa sa tačnošću od 76.62% i AUC-ROC vrednošću od 0.8226. Rezultati su uporedivi sa literaturom, a metodološki doprinos rada ogleda se u pažljivoj pripremi podataka bez curenja informacija, automatskoj hiperparametarskoj optimizaciji i medicinski opravdanom tretmanu nebalansiranosti klasa kroz optimizaciju praga odluke. Buduća istraživanja mogla bi uključiti poređenje sa Random Forest-om i XGBoost-om te validaciju na većim i demografski raznovrsnijim skupovima podataka.

---

## Pokretanje projekta

### Preduslovi

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn optuna shap
```

### Reprodukcija rezultata

1. Kloniraj repozitorijum:
```bash
git clone https://github.com/dragutinovicj/predvidjanje-dijabetesa.git
cd predvidjanje-dijabetesa
```

2. Preuzmi dataset sa Kaggle-a i postavi `diabetes.csv` u isti direktorijum kao notebook

3. Otvori i pokreni notebook u Google Colab-u ili lokalno:
```bash
jupyter notebook Predviđanje_dijabetesa.ipynb
```

---

## Struktura repozitorijuma
├── Predviđanje_dijabetesa.ipynb   # Glavni notebook
├── diabetes.csv                    # Skup podataka
└── README.md                       # Opis projekta

---

## Licenca

MIT License — slobodno korišćenje, modifikacija i distribucija uz navođenje izvora.
