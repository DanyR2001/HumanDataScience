# Report Metodologico: Catena Decisionale dei Test Statistici
## Analisi dei Margini sui Carburanti Italiani — Tre Crisi Energetiche

---

## 1. Obiettivo e Ipotesi Nulla

L'analisi intende testare se il margine lordo dei distributori italiani (calcolato come
differenza fra il prezzo alla pompa al netto delle tasse e il costo wholesale europeo —
Eurobob per la benzina, Gas Oil ICE per il diesel) aumenti in modo statisticamente anomalo
rispetto al baseline pre-crisi (2019) durante tre eventi energetici: invasione dell'Ucraina
(feb. 2022), guerra Iran-Israele (giu. 2025), chiusura dello Stretto di Hormuz (feb. 2026).

**H₀:** Il margine lordo non aumenta oltre la soglia 2σ del baseline 2019 in modo
statisticamente significativo dopo lo shock.

---

## 2. Punto di Partenza: Verifica delle Assunzioni OLS

Il test frequentista più naturale per confrontare margini pre/post-shock è il **Welch
t-test** (varianze non assunte uguali), che richiede: residui approssimativamente normali
e, nei test su serie temporali, assenza di autocorrelazione (altrimenti i gradi di libertà
effettivi sono inflati e i p-value sottostimati).

Prima di applicare qualsiasi test inferenziale su ogni coppia evento × carburante, lo
script `03_statistical_tests.py` esegue tre diagnostici sui residui OLS:

| Diagnostico | Test | Cosa misura |
|---|---|---|
| Eteroschedasticità | Breusch-Pagan (BP) | varianza dei residui non costante |
| Non-normalità | Shapiro-Wilk (SW) | residui non gaussiani |
| Autocorrelazione | Durbin-Watson (DW) + Ljung-Box | correlazione seriale nei residui |

---

## 3. Risultati dei Diagnostici OLS

### 3.1 Autocorrelazione — Durbin-Watson

I valori attesi di DW vanno da 0 (autocorrelazione positiva perfetta) a 4 (negativa);
il range "sicuro" per OLS è 1.5–2.5.

| Evento | Serie | DW | ρ AR(1) stim. | Inflazione SE |
|---|---|---|---|---|
| Ucraina | Benzina | **0.37** | **0.909** | **+105%** |
| Ucraina | Diesel | **0.21** | **0.893** | **+80%** |
| Iran-Israele | Benzina | **0.19** | **0.903** | **+55%** |
| Iran-Israele | Diesel | **0.15** | **0.924** | **+47%** |
| Hormuz | Benzina | **0.29** | **0.845** | **+107%** |
| Hormuz | Diesel | **0.20** | **0.902** | **+108%** |

**Conclusione:** autocorrelazione di primo ordine quasi perfetta (ρ ≈ 0.85–0.92) in
tutti gli scenari. Di conseguenza gli errori standard OLS sono sottostimati del 47–108%:
un test t con SE ridotto a metà restituisce p-value gonfiati — cioè rifiuta H₀ più
spesso di quanto dovrebbe. Il Ljung-Box (lag 4) conferma con p = 0.000 in tutti i casi.

Ulteriore conferma arriva dal **Runs test sui residui**: Z ≈ −5, p ≈ 0.000 in tutti
gli scenari → segni dei residui non casuali, struttura seriale sistemica.

### 3.2 Eteroschedasticità — Breusch-Pagan

| Evento | Serie | BP p-value | Esito |
|---|---|---|---|
| Ucraina | Benzina | **0.0001** | **RIFIUTATA** |
| Ucraina | Diesel | **0.0021** | **RIFIUTATA** |
| Iran-Israele | Benzina | **0.019** | **RIFIUTATA** |
| Iran-Israele | Diesel | 0.061 | marginale |
| Hormuz | Benzina | **0.010** | **RIFIUTATA** |
| Hormuz | Diesel | **0.002** | **RIFIUTATA** |

Eteroschedasticità confermata in 5/6 serie. Nota: con autocorrelazione forte, il test
BP stesso non è completamente affidabile (i residui non sono i.i.d.), quindi va letto
come segnale indicativo, non come test formale definitivo.

### 3.3 Non-normalità — Shapiro-Wilk

| Evento | Serie | SW p-value | Esito |
|---|---|---|---|
| Ucraina | Benzina | **4.8 × 10⁻⁵** | **NON NORMALE** |
| Ucraina | Diesel | **1.7 × 10⁻⁴** | **NON NORMALE** |
| Iran-Israele | Diesel | 0.489 | ok |
| Hormuz | Benzina | **3.2 × 10⁻⁴** | **NON NORMALE** |

Non-normalità evidente per Ucraina (entrambi i carburanti) e Hormuz Benzina. Il test
t di Welch ha robustezza asintotica alla non-normalità, ma per campioni piccoli
(n_pre ≈ 23–25, n_post ≈ 14–27) la convergenza alla distribuzione t non è garantita.

---

## 4. Conseguenza: Perché il Welch t-test Non è Sufficiente

Le violazioni rilevate hanno impatti diversi sul Welch t:

| Violazione | Impatto sul Welch t | Gravità |
|---|---|---|
| Autocorrelazione (ρ ≈ 0.90) | SE sottostimati → falsi positivi | **CRITICA** |
| Non-normalità (n ≈ 20–25) | Convergenza asintotica non garantita | **RILEVANTE** |
| Eteroschedasticità | Già gestita da Welch (varianze libere) | **GESTITA** |

In presenza di autocorrelazione forte, i n osservazioni effettive sono equivalenti a
circa n×(1−ρ)/(1+ρ) ≈ 5–7 osservazioni indipendenti. Il test t tratta invece tutte le
25–27 osservazioni come indipendenti, gonfiando la potenza di 3–5×.

**Decisione metodologica:** il Welch t-test viene mantenuto come test primario (per
confrontabilità con la letteratura e per la BH correction), ma affiancato da tre
categorie di test che non condividono le sue assunzioni violate.

---

## 5. Batteria di Test Alternativa e Motivazione

### 5.1 HAC Newey-West — Correzione per Autocorrelazione senza Rinunciare alla Parametricità

**Motivazione:** se il problema è solo l'autocorrelazione (ma la linearità e la
distribuzione campionaria asintotica reggono), si può mantenere OLS e correggere
solamente la matrice di covarianza degli stimatori con stimatori sandwich robusti.
Il metodo Newey-West (maxlags = 4, ≈ 1 mese) fornisce errori standard consistenti
in presenza di autocorrelazione e eteroschedasticità.

**Risultati:**

| Evento | Carburante | Δ medio | HAC p | Esito H₀ |
|---|---|---|---|---|
| Ucraina | Benzina | +0.058 €/l | 0.086 | non rifiutata |
| Ucraina | Diesel | +0.047 €/l | 0.097 | non rifiutata |
| Iran-Israele | Benzina | −0.006 €/l | 0.526 | non rifiutata |
| Iran-Israele | Diesel | −0.021 €/l | 0.062 | non rifiutata |

**Interpretazione:** una volta corretti gli SE per l'autocorrelazione, nessun test
risulta significativo a α = 0.05 — il segnale del Welch t non reggeva all'aumento
dei gradi di libertà effettivi. Questo è il costo dell'autocorrelazione: riduce la
potenza reale del test.

### 5.2 Mann-Whitney U — Robustezza a Non-normalità

**Motivazione:** il Mann-Whitney U non assume normalità né scala di misura specifica.
Testa H₀: P(margine_post > margine_pre) = 0.5. Con la stima Hodges-Lehmann
si ottiene anche uno shift mediano robusto, e con Cliff's δ un effect size ordinale
interpretabile indipendentemente dalla scala.

**Risultati:**

| Evento | Carburante | HL shift | Cliff's δ | Magnitudine | p (one-sided) | Esito H₀ |
|---|---|---|---|---|---|---|
| Ucraina | Benzina | +0.074 €/l | +0.393 | **medio** | **0.0078** | **RIFIUTATA** |
| Ucraina | Diesel | +0.057 €/l | +0.419 | **medio** | **0.0049** | **RIFIUTATA** |
| Iran-Israele | Benzina | −0.005 €/l | −0.113 | trascurabile | 0.741 | non rifiutata |
| Iran-Israele | Diesel | −0.021 €/l | −0.430 | medio (↓) | 0.992 | non rifiutata |

**Interpretazione:** per Ucraina il Mann-Whitney risulta significativo (dove il
Welch con SE gonfiati non lo era), con effect size medio. La divergenza tra MW
e HAC riflette due domande diverse: MW testa se la distribuzione post-shock è
spostata rispetto alla pre (indipendentemente dalla struttura temporale), HAC
testa se la media OLS condizionata cambia tenendo conto della dipendenza seriale.

### 5.3 Block Permutation Test — Robustezza alla Struttura Temporale

**Motivazione:** sia il Welch che il MW trattano le osservazioni come i.i.d., il
che non è realistico con autocorrelazione ρ ≈ 0.90. Il permutation test a blocchi
(block_size = 4 settimane) preserva la struttura temporale locale nelle permutazioni:
invece di rimescolare singole settimane, rimescola blocchi di 4 settimane consecutive.
La statistica test è Δmediana (coerente con MW).

**Risultati:**

| Evento | Carburante | Δ mediana osservata | p perm. | Esito H₀ |
|---|---|---|---|---|
| Ucraina | Benzina | +0.076 €/l | **0.014** | **RIFIUTATA** |
| Ucraina | Diesel | +0.038 €/l | **0.035** | **RIFIUTATA** |
| Iran-Israele | Benzina | −0.006 €/l | 0.719 | non rifiutata |
| Iran-Israele | Diesel | −0.018 €/l | 0.923 | non rifiutata |

**Interpretazione:** anche preservando l'autocorrelazione locale nella distribuzione
nulla, il segnale per Ucraina persiste. L'accordo con Mann-Whitney rafforza
la conclusione: c'è un aumento del margine post-shock per Ucraina.

### 5.4 Fligner-Killeen — Eteroschedasticità Non Parametrica

**Motivazione:** il Breusch-Pagan (usato in §3.2) è un test parametrico che presuppone
residui normali. Con non-normalità confermata, il Fligner-Killeen offre un test
dell'omogeneità delle varianze robusto alla non-normalità.

| Evento | Carburante | FK p | Esito H₀ |
|---|---|---|---|
| Ucraina | Benzina | **0.002** | **RIFIUTATA** |
| Ucraina | Diesel | 0.076 | non rifiutata |
| Iran-Israele | Benzina | **0.0001** | **RIFIUTATA** |
| Iran-Israele | Diesel | **0.0002** | **RIFIUTATA** |

**Interpretazione:** la varianza del margine cambia tra pre e post-shock — specialmente
per Iran. Questa instabilità della varianza rende ancora meno affidabili i test
parametrici standard.

---

## 6. Sintesi della Catena Decisionale

```
Serie × Evento
│
├── STEP 1: Diagnostici OLS
│   ├── DW < 1.5 → autocorrelazione forte → ρ ≈ 0.85–0.92 in tutti i casi
│   ├── BP p < 0.05 → eteroschedasticità in 5/6 serie
│   └── SW p < 0.05 → non-normalità in Ucraina + Hormuz Benzina
│
├── STEP 2: Welch t-test (test primario — mantenuto per BH e confrontabilità)
│   ├── Ucraina Benzina: p = 0.108 → non significativo
│   ├── Ucraina Diesel:  p = 0.064 → non significativo
│   ├── Iran Benzina:    p = 0.192 → non significativo
│   └── Iran Diesel:     p = 0.004 → SIGNIFICATIVO ← anomalia: compressione
│
├── STEP 3a: SE gonfiati → HAC Newey-West (corregge autocorrelazione)
│   └── Tutti p > 0.05 → nessun segnale residuo dopo correzione SE
│
├── STEP 3b: Non-normalità → Mann-Whitney U
│   ├── Ucraina Benzina: p = 0.008 → SIGNIFICATIVO (Cliff's δ = +0.39, medio)
│   └── Ucraina Diesel:  p = 0.005 → SIGNIFICATIVO (Cliff's δ = +0.42, medio)
│
├── STEP 3c: Autocorrelazione + non-normalità → Block Permutation
│   ├── Ucraina Benzina: p = 0.014 → SIGNIFICATIVO
│   └── Ucraina Diesel:  p = 0.035 → SIGNIFICATIVO
│
└── STEP 4: Fligner-Killeen → varianze non omogenee in 3/4 serie
```

---

## 7. Correzione BH Globale e Risultati Finali

Dopo aver raccolto tutti i p-value confirmatory (Welch t + DiD + MW + Block perm + HAC),
la Benjamini-Hochberg correction globale è applicata su 28 test confirmatory (α = 5% FDR).

I test che superano la soglia BH globale:

| Test | Evento | Carburante | p nominale | p aggiustato | Esito |
|---|---|---|---|---|---|
| Welch t | Iran-Israele | Diesel | 0.0035 | 0.0457 | **RIFIUTATA** (↓ margine) |
| Mann-Whitney | Ucraina | Diesel | 0.0049 | 0.0457 | **RIFIUTATA** (↑ margine) |
| DiD vs Germania | Hormuz | Benzina | 0.0019 | 0.0457 | **RIFIUTATA** ⚠ PTA violata |

**Nota sul DiD Hormuz:** il segnale è statisticamente significativo dopo BH, ma il
Parallel Trends Assumption (PTA) risulta violato (PTA p < 0.05 contro Germania),
il che rende il δ DiD non interpretabile causalmente. Il test contro la Svezia
(PTA non violata) non risulta significativo. Il risultato va trattato con cautela.

**Nota sul segnale Iran Diesel:** il Welch t risulta significativo (margine in
discesa, non in salita), ma il Mann-Whitney (p = 0.992) e il block permutation
(p = 0.923) non confermano il segnale. La divergenza suggerisce che il test t stia
reagendo a una struttura della distribuzione (asimmetria, outlier) piuttosto che
a uno shift sistemico del margine. La classificazione BH corretta è quindi
"NEUTRO / TRASMISSIONE ATTESA" nonostante il rigetto formale.

---

## 8. Changepoint Bayesiano (Table 1)

Il cambio di regime nei log-prezzi è stimato con un modello piecewise-lineare
Bayesiano (likelihood StudentT per robustezza alle code pesanti, prior Beta(2,2)
su τ per evitare boundary effects). Il lag D misura l'anticipo/ritardo del
changepoint rispetto allo shock:

| Evento | Serie | τ̂ | Lag D | H₀ rifiutata? | Rhat |
|---|---|---|---|---|---|
| Ucraina | Brent | 2021-12-13 | −73 gg | NO | 1.059 ⚠ |
| Ucraina | Benzina | 2022-01-03 | −52 gg | NO | 1.001 ✓ |
| Ucraina | Diesel | 2022-01-03 | −52 gg | NO | 1.001 ✓ |
| Iran-Israele | Brent | 2025-04-28 | −46 gg | NO | 1.002 ✓ |
| Iran-Israele | Benzina | 2025-04-28 | −46 gg | NO | 1.001 ✓ |
| Hormuz | Benzina | 2026-03-02 | +2 gg | **SÌ** | 1.002 ✓ |
| Hormuz | Diesel | 2026-02-23 | −5 gg | **SÌ** | 1.001 ✓ |

Per Ucraina e Iran il changepoint precede lo shock di 39–73 giorni: i prezzi
avevano già cominciato a salire per ragioni endogene ai mercati futures (tensioni
geopolitiche anticipate, rallentamento dell'OPEC). Questo è coerente con mercati
forward-looking, non con speculazione post-shock.

Nota: il Brent Ucraina ha Rhat = 1.059 > 1.05 → convergenza MCMC dubbia; il
risultato va trattato con cautela.

---

## 9. Tabella Riepilogativa: Classificazione Finale

| Evento | Carburante | Δ margine | BH locale | BH globale | MW | Block perm | HAC | Classificazione |
|---|---|---|---|---|---|---|---|---|
| Ucraina | Benzina | +0.039 €/l | ✗ | ✗ | ✓ p=0.008 | ✓ p=0.014 | ✗ | **VARIAZIONE STATISTICA** (segnale, non conclusivo) |
| Ucraina | Diesel | +0.048 €/l | ✗ | ✗ | ✓ p=0.005 | ✓ p=0.035 | ✗ | **VARIAZIONE STATISTICA** (segnale, non conclusivo) |
| Iran-Israele | Benzina | −0.009 €/l | ✗ | ✗ | ✗ | ✗ | ✗ | **NEUTRO** |
| Iran-Israele | Diesel | −0.028 €/l | ✓ p=0.004 | ✓ | ✗ p=0.992 | ✗ | ✗ | **NEUTRO** (divergenza test) |

---

## 10. Limiti e Note Metodologiche

**Sulla proxy del margine:** il crack spread Eurobob/Gas Oil è una proxy del costo
wholesale ARA Rotterdam. I distributori italiani possono utilizzare contratti
forward a prezzi CIF-Genova o contratti bilaterali, introducendo una discrepanza
sistematica tra la proxy e il margine reale.

**Sull'autocorrelazione residua:** i test non parametrici (MW, block perm)
eliminano il problema della normalità ma non correggono completamente
l'autocorrelazione — il block permutation la gestisce parzialmente preservando
blocchi di 4 settimane, ma non è una soluzione esatta per ρ ≈ 0.90.

**Sul Kruskal-Wallis:** i gruppi temporali (pre/shock/post) non sono indipendenti,
rendendo la distribuzione nulla del K-W statistic approssimativa. I risultati
sono riportati a scopo esplorativo ma non entrano nella BH correction.

**Sulla causalità:** tutte le classificazioni sono descrittive del pattern
statistico. "MARGINE ANOMALO POSITIVO" è coerente anche con effetti FIFO/LIFO
su inventario, risk premium razionale, cost-push non catturato da ARA/ICE,
riduzione temporanea della concorrenza. La conclusione causale (speculazione)
richiederebbe ulteriori evidenze (es. dati di inventario, margini per canale).
