# Report Metodologico: Catena Decisionale dei Test Statistici
## Analisi dei Margini sui Carburanti Italiani — Tre Crisi Energetiche

---

## 1. Obiettivo e Ipotesi Nulla

L'analisi intende testare se il margine lordo dei distributori italiani — calcolato come
differenza fra il prezzo alla pompa al netto delle tasse e il costo wholesale europeo
(Eurobob per la benzina, Gas Oil ICE per il diesel, entrambi convertiti in EUR/litro) —
aumenti in modo statisticamente anomalo rispetto al baseline pre-crisi (2019) durante
tre eventi energetici: invasione dell'Ucraina (feb. 2022), guerra Iran-Israele (giu. 2025),
chiusura dello Stretto di Hormuz (feb. 2026).

**H₀:** μ_post = μ_2019 (il margine lordo medio nel periodo post-shock è uguale alla
media del baseline 2019). Il test è **unilaterale superiore** (one-sided upper):
interessa solo se il margine è salito, non se è sceso.

**Baseline:** 2019 full year (52 settimane). Valori di riferimento:
- Benzina: μ₂₀₁₉ = 0.168 EUR/L, σ = 0.019, soglia 2σ = 0.038 EUR/L
- Diesel: μ₂₀₁₉ = 0.149 EUR/L, σ = 0.018, soglia 2σ = 0.037 EUR/L

Dataset: 381 settimane, 2019-01-07 – 2026-04-20.

---

## 2. Punto di Partenza: Verifica delle Assunzioni OLS (Script 02)

Prima di applicare qualsiasi test inferenziale su ogni coppia evento × carburante, lo
script `02_changepoint.py` produce tre diagnostici sui residui della regressione
piecewise-lineare applicata ai *log-prezzi*. Questi diagnostici guidano la scelta
della batteria di test di script 03.

| Diagnostico | Test | Cosa misura |
|---|---|---|
| Autocorrelazione | Durbin-Watson (DW) | correlazione seriale nei residui |
| Non-normalità | Shapiro-Wilk (SW) | residui non gaussiani |
| Eteroschedasticità | Breusch-Pagan (BP) | varianza dei residui non costante |

---

## 3. Risultati dei Diagnostici OLS

### 3.1 Autocorrelazione — Durbin-Watson

I valori attesi di DW vanno da 0 (autocorrelazione positiva perfetta) a 4 (negativa);
il range "sicuro" per OLS è 1.5–2.5.

| Evento | Serie | DW | Esito |
|---|---|---|---|
| Ucraina | Brent | 0.31 | **AUTOCORR.** |
| Ucraina | Benzina | 0.37 | **AUTOCORR.** |
| Ucraina | Diesel | 0.29 | **AUTOCORR.** |
| Iran-Israele | Brent | 1.20 | **AUTOCORR.** |
| Iran-Israele | Benzina | 0.42 | **AUTOCORR.** |
| Iran-Israele | Diesel | 0.33 | **AUTOCORR.** |
| Hormuz | Brent | 1.09 | **AUTOCORR.** |
| Hormuz | Benzina | 1.32 | **AUTOCORR.** |
| Hormuz | Diesel | 0.93 | **AUTOCORR.** |

**Conclusione:** autocorrelazione sistematica in tutte le serie (DW < 1.5 in 9/9 casi).
I valori DW più bassi (0.29–0.42) corrispondono a ρ AR(1) stimato di ≈ 0.85–0.90.
Con ρ ≈ 0.90, le 25–27 osservazioni post-shock non valgono 25 indipendenti ma
l'equivalente di circa 5–7: il test t tratta invece tutte come indipendenti,
gonfiando la potenza di 3–5 volte.

Nota: i DW qui si riferiscono ai residui OLS sui *log-prezzi* (finestra evento × serie),
non al crack spread. I diagnostici OLS sul crack spread mostrano pattern analoghi.

### 3.2 Non-normalità — Shapiro-Wilk

| Evento | Serie | SW p-value | Esito |
|---|---|---|---|
| Ucraina | Benzina | 0.0000 | **NON NORMALE** |
| Ucraina | Diesel | 0.0002 | **NON NORMALE** |
| Iran-Israele | Benzina | 0.8690 | ok |
| Iran-Israele | Diesel | 0.4891 | ok |
| Hormuz | Benzina | 0.0003 | **NON NORMALE** |
| Hormuz | Diesel | 0.6817 | ok |

Non-normalità evidente per Ucraina (entrambi i carburanti) e Hormuz Benzina. Il test t
di Welch ha robustezza asintotica alla non-normalità, ma per campioni piccoli
(n_pre ≈ 23–25, n_post ≈ 7–27) la convergenza alla distribuzione t non è garantita.

### 3.3 Eteroschedasticità — Breusch-Pagan

| Evento | Serie | BP p-value | Esito |
|---|---|---|---|
| Ucraina | Benzina | 0.3493 | ok |
| Ucraina | Diesel | 0.3684 | ok |
| Iran-Israele | Benzina | 0.1231 | ok |
| Iran-Israele | Diesel | **0.0211** | **ETEROSC.** |
| Hormuz | Benzina | **0.0098** | **ETEROSC.** |
| Hormuz | Diesel | 0.1360 | ok |

Eteroschedasticità confermata in 2/6 serie principali (Iran Diesel, Hormuz Benzina).
Nota: con autocorrelazione forte, il BP è esso stesso non completamente affidabile
(i residui non sono i.i.d.), quindi va letto come segnale indicativo.

---

## 4. Conseguenza: Perché il Welch t-test Non è Sufficiente

| Violazione | Impatto sul Welch t | Gravità |
|---|---|---|
| Autocorrelazione (DW < 1.5) | SE sottostimati → falsi positivi | **CRITICA** |
| Non-normalità (Ucraina, Hormuz benz.) | Convergenza asintotica non garantita (n piccolo) | **RILEVANTE** |
| Eteroschedasticità | Parzialmente gestita da Welch | **GESTITA** |

**Decisione metodologica:** il Welch t-test viene mantenuto come test primario (per
confrontabilità con la letteratura e per la BH correction), ma affiancato da tre
categorie di test che non condividono le sue assunzioni violate.

---

## 5. Batteria di Test e Motivazione

### 5.1 Welch t-test a campione singolo (test primario)

**Motivazione:** test one-sample che confronta μ_post con il valore fisso μ_2019.
È one-sided upper: H₀: μ_post ≤ μ_2019 vs H₁: μ_post > μ_2019. Usando μ_2019
come valore fisso (non come campione separato), elimina la variabilità della baseline
dal calcolo degli SE.

**Risultati:**

| Evento | Carburante | n_pre | n_post | δ_vs_2019 | t | p (one-sided) |
|---|---|---|---|---|---|---|
| Ucraina | Benzina | 25 | 27 | +0.089 EUR/L | 4.490 | 0.0001 *** |
| Ucraina | Diesel | 25 | 27 | +0.073 EUR/L | 3.531 | 0.0008 *** |
| Iran-Israele | Benzina | 23 | 20 | +0.079 EUR/L | 20.484 | 0.0000 *** |
| Iran-Israele | Diesel | 23 | 20 | +0.060 EUR/L | 9.317 | 0.0000 *** |

Hormuz: escluso dalla BH (dati preliminari, n_post = 7).

**Nota critica:** con autocorrelazione, questi p-value sono sottostimati — cioè H₀
viene rigettata più spesso di quanto dovrebbe. Il confronto con i test 5.3 e 5.4
(che gestiscono l'autocorrelazione) permette di valutare la dimensione di questo bias.

### 5.2 Mann-Whitney U — Robustezza a Non-normalità

**Motivazione:** test non parametrico su ranghi. Non assume normalità né varianza
omogenea. Il confronto è costruito come **post-shock vs distribuzione 2019 intera**
(n_2019 = 52), non vs finestra pre-shock dello stesso evento. Questa scelta risolve
la potenziale contaminazione del "pre" (già anomalo in anticipo rispetto allo shock)
e allinea il test con H₀ riferita a μ_2019.

**Statistiche prodotte:** AUC (probabilità P[post > 2019]), Hodges-Lehmann (shift
mediano robusto), Cliff's delta (effect size ordinale, −1 a +1).

**Risultati:**

| Evento | Carburante | HL (EUR/L) | Cliff's δ | Magnitudine | p (one-sided) |
|---|---|---|---|---|---|
| Ucraina | Benzina | +0.107 | +0.504 | **grande** | 0.0001 *** |
| Ucraina | Diesel | +0.075 | +0.550 | **grande** | 0.0000 *** |
| Iran-Israele | Benzina | +0.078 | +1.000 | **grande** | 0.0000 *** |
| Iran-Israele | Diesel | +0.067 | +0.881 | **grande** | 0.0000 *** |

**Nota per Iran-Israele:** il MW rigetta (margine post > 2019) ma il δ_locale è
negativo (margine post < pre dello stesso evento). Queste non sono affermazioni
contraddittorie: il MW dice che il livello assoluto del margine post-shock è sopra il
2019; il δ_locale dice che quel livello è sceso rispetto alla finestra pre dello stesso
evento (che era già anomalamente alta). La classificazione finale riflette entrambi
gli aspetti.

### 5.3 Block Permutation Test — Robustezza alla Struttura Temporale

**Motivazione:** sia il Welch che il MW trattano le osservazioni come i.i.d. Il block
permutation test (block_size = 4 settimane) preserva la struttura temporale locale
nelle permutazioni: rimescola blocchi di 4 settimane consecutive. È il test più
robusto all'autocorrelazione, a costo di potenza ridotta rispetto ai test parametrici.

**Dual split implementation:** il test è implementato in due varianti:

| Variante | Split date | Ruolo | Entra in BH? |
|---|---|---|---|
| **Principale** | τ_price (da script 02) | Esogeno al margine; pre window pulita | ✓ Sì |
| **Robustness** | τ_margin (da script 03) | Endogeno; cattura rottura del margine | ✗ No |

**Risultati (split principale τ_price):**

| Evento | Carburante | τ_price | Δ mediano | p | Esito BH |
|---|---|---|---|---|---|
| Ucraina | Benzina | 2022-01-03 | +0.038 EUR/L | 0.125 | n.s. |
| Ucraina | Diesel | 2022-01-03 | +0.011 EUR/L | 0.365 | n.s. |
| Iran-Israele | Benzina | 2025-04-28 | −0.017 EUR/L | 0.913 | n.s. |
| Iran-Israele | Diesel | 2025-05-05 | −0.022 EUR/L | 0.964 | n.s. |

**Risultati (split robustness τ_margin):**

| Evento | Carburante | τ_margin | Δ mediano | p |
|---|---|---|---|---|
| Ucraina | Benzina | 2022-03-14 | +0.095 EUR/L | 0.008 ** |
| Ucraina | Diesel | 2022-03-14 | +0.060 EUR/L | 0.009 ** |
| Iran-Israele | Benzina | 2025-04-21 | −0.016 EUR/L | 0.930 n.s. |
| Iran-Israele | Diesel | 2025-06-02 | −0.022 EUR/L | 0.973 n.s. |

**Tensione Ucraina (τ_lag = +70 giorni):** τ_price = 3 gen 2022, τ_margin = 14 mar
2022. Con split τ_price, le 10 settimane di transizione graduale del margine finiscono
nel "post" e diluiscono il delta mediano. Con τ_margin il segnale è molto più netto.
Entrambi i risultati sono informativi: il primo risponde a "c'è stato un salto brusco
del margine subito dopo il segnale di prezzo?" (no); il secondo a "c'è stata una
rottura strutturale nel margine?" (sì, ma con 10 settimane di ritardo).

### 5.4 HAC Newey-West — Correzione Parametrica per Autocorrelazione

**Motivazione:** mantiene la struttura OLS parametrica ma sostituisce la matrice di
covarianza ordinaria con uno stimatore Newey-West consistente ad autocorrelazione e
eteroschedasticità (maxlags = 4). Tratta la dipendenza seriale in modo diverso dal
block permutation: invece di preservarla nelle permutazioni, la incorporate nell'errore
standard stimato. I due approcci sono complementari.

**Stessa dual split implementation del blocco 5.3.**

**Risultati (split principale τ_price):**

| Evento | Carburante | δ HAC | p HAC | Esito BH |
|---|---|---|---|---|
| Ucraina | Benzina | +0.041 EUR/L | 0.178 | n.s. |
| Ucraina | Diesel | +0.019 EUR/L | 0.496 | n.s. |
| Iran-Israele | Benzina | −0.019 EUR/L | 0.067 | n.s. |
| Iran-Israele | Diesel | −0.027 EUR/L | 0.042 * | non rigettato (BH adj.) |

**Risultati (split robustness τ_margin):**

| Evento | Carburante | δ HAC | p HAC |
|---|---|---|---|
| Ucraina | Benzina | +0.077 EUR/L | 0.039 * |
| Ucraina | Diesel | +0.074 EUR/L | 0.014 * |
| Iran-Israele | Benzina | −0.020 EUR/L | 0.058 n.s. |
| Iran-Israele | Diesel | −0.028 EUR/L | 0.023 * |

Il HAC Iran Diesel con τ_price dà p = 0.042 (nominalmente significativo) ma
p_adj_BH = 0.113, non rigettato dopo correzione globale.

### 5.5 Il τ_margin: changepoint del margine lordo

Oltre al changepoint del prezzo (τ_price, da script 02), lo script 03 stima un
changepoint del crack spread stesso (τ_margin) con il metodo brute-force sulla
statistica di Welch t rolling. Il confronto tra i due produce una classificazione
del comportamento temporale del margine:

| Evento | Carburante | τ_price | τ_margin | τ_lag | Tipo |
|---|---|---|---|---|---|
| Ucraina | Benzina | 2022-01-03 | 2022-03-14 | +70 gg | **REATTIVO** |
| Ucraina | Diesel | 2022-01-03 | 2022-03-14 | +70 gg | **REATTIVO** |
| Iran-Israele | Benzina | 2025-04-28 | 2025-04-21 | −7 gg | **SINCRONO** |
| Iran-Israele | Diesel | 2025-05-05 | 2025-06-02 | +28 gg | **REATTIVO** |
| Hormuz (prel.) | Benzina | 2026-03-02 | 2025-11-24 | −98 gg | **ANTICIPATORIO** |
| Hormuz (prel.) | Diesel | 2026-02-23 | 2026-01-05 | −49 gg | **ANTICIPATORIO** |

**Interpretazione economica:**
- REATTIVO: coerente con cost pass-through graduale; il distributore aggiusta il
  prezzo man mano che assorbe il costo wholesale più elevato.
- SINCRONO: espansione del margine contestuale al movimento di prezzo; il distributore
  aggiusta il margine insieme al prezzo, non in ritardo.
- ANTICIPATORIO: il margine si espande *prima* che il prezzo wholesale salga. Per
  Hormuz, con soli 7 dati post-shock, il risultato è preliminare e da non interpretare
  causalmente.

---

## 6. Sintesi della Catena Decisionale

```
Serie × Evento
│
├── STEP 1 [Script 02]: Diagnostici OLS sui log-prezzi
│   ├── DW < 1.5 in 9/9 serie → autocorrelazione sistematica
│   ├── SW p < 0.05 in 3/9 serie (Ucraina benz/dies, Hormuz benz) → non-normalità
│   └── BP p < 0.05 in 2/9 serie (Iran dies, Hormuz benz) → eteroschedasticità
│
├── STEP 2 [Script 03]: Welch t-test (test primario vs μ_2019, one-sided upper)
│   ├── Ucraina Benzina: p = 0.0001 *** → RIFIUTA H₀
│   ├── Ucraina Diesel:  p = 0.0008 *** → RIFIUTA H₀
│   ├── Iran Benzina:    p = 0.0000 *** → RIFIUTA H₀
│   └── Iran Diesel:     p = 0.0000 *** → RIFIUTA H₀
│
├── STEP 3 [Script 03]: Mann-Whitney U (post vs 2019 baseline, n_2019=52)
│   ├── Ucraina Benzina: p = 0.0001, Cliff's δ = +0.504 [grande] → RIFIUTA
│   ├── Ucraina Diesel:  p = 0.0000, Cliff's δ = +0.550 [grande] → RIFIUTA
│   ├── Iran Benzina:    p = 0.0000, Cliff's δ = +1.000 [grande] → RIFIUTA
│   └── Iran Diesel:     p = 0.0000, Cliff's δ = +0.881 [grande] → RIFIUTA
│
├── STEP 4 [Script 03]: Block Permutation — split PRINCIPALE (τ_price)
│   ├── Ucraina Benzina: p = 0.125 n.s. (gap 70gg diluisce delta)
│   ├── Ucraina Diesel:  p = 0.365 n.s.
│   ├── Iran Benzina:    p = 0.913 n.s. (δ negativo)
│   └── Iran Diesel:     p = 0.964 n.s. (δ negativo)
│
├── STEP 4bis [Script 03]: Block Permutation — split ROBUSTNESS (τ_margin)
│   ├── Ucraina Benzina: p = 0.008 ** → segnale confermato a τ_margin
│   ├── Ucraina Diesel:  p = 0.009 ** → segnale confermato a τ_margin
│   ├── Iran Benzina:    p = 0.930 n.s. → robustness: nessun segnale
│   └── Iran Diesel:     p = 0.973 n.s. → robustness: nessun segnale
│
├── STEP 5 [Script 03]: HAC Newey-West — split PRINCIPALE (τ_price)
│   ├── Ucraina Benzina: p = 0.178 n.s.
│   ├── Ucraina Diesel:  p = 0.496 n.s.
│   ├── Iran Benzina:    p = 0.067 n.s.
│   └── Iran Diesel:     p = 0.042 * (ma p_adj_BH = 0.113 → non rigettato)
│
└── STEP 6 [Script 04]: DiD IT vs DE e SE
    ├── Tutti p > 0.05 con PTA non violata (6/12 casi)
    └── → anomalia non specifica all'Italia
```

---

## 7. Correzione BH Globale e Risultati Finali

Tutti i p-value confirmatory sono raccolti in due file (script 03 → confirmatory_pvalues.csv,
script 04 → auxiliary_pvalues.csv) e passati a script 05 per la BH globale.

| Categoria test | Fonte | n test |
|---|---|---|
| Welch t (vs μ_2019) | script 03 | 4 |
| Mann-Whitney (post vs 2019) | script 03 | 4 |
| Block permutation (split τ_price, principale) | script 03 | 4 |
| HAC Newey-West (split τ_price, principale) | script 03 | 4 |
| DiD IT vs DE/SE (2 paesi × 2 carburanti × 2 eventi) | script 04 | 8 |
| **Totale famiglia BH** | | **24** |

Test robustness (split τ_margin), Granger, R&F non entrano nella famiglia BH.
Hormuz escluso dalla BH per dati insufficienti (n_post = 7).

**Risultati BH globale (α = 5% FDR):**

| Test | Evento | Carburante | p nom. | p adj. | Esito |
|---|---|---|---|---|---|
| Mann-Whitney | Ucraina | Diesel | 0.0000 | 0.0000 | **RIFIUTATA** |
| Mann-Whitney | Iran-Israele | Benzina | 0.0000 | 0.0000 | **RIFIUTATA** |
| Mann-Whitney | Iran-Israele | Diesel | 0.0000 | 0.0000 | **RIFIUTATA** |
| Welch t | Iran-Israele | Benzina | 0.0000 | 0.0000 | **RIFIUTATA** |
| Welch t | Iran-Israele | Diesel | 0.0000 | 0.0000 | **RIFIUTATA** |
| Welch t | Ucraina | Benzina | 0.0001 | 0.0003 | **RIFIUTATA** |
| Mann-Whitney | Ucraina | Benzina | 0.0001 | 0.0003 | **RIFIUTATA** |
| Welch t | Ucraina | Diesel | 0.0008 | 0.0024 | **RIFIUTATA** |
| HAC | Iran-Israele | Diesel | 0.0422 | 0.1125 | non rigettata |
| Tutti BlockPerm (principale) | — | — | 0.125–0.964 | >0.25 | non rigettate |
| Tutti DiD | — | — | 0.052–0.967 | >0.12 | non rigettate |

**Sommario per famiglia:**
- Welch t: **4/4** rigettati
- Mann-Whitney: **4/4** rigettati
- Block permutation (principale): 0/4 rigettati
- HAC (principale): 0/4 rigettati
- DiD: 0/8 rigettati

---

## 8. Interpretazione del Quadro Complessivo

### 8.1 Convergenza Welch t + MW (livello assoluto vs 2019)

Entrambi i test di confronto assoluto con il 2019 rigettano in tutti i 4 casi non
preliminari. Questo significa che il livello del margine nel periodo post-shock è
sistematicamente e significativamente superiore al 2019, in modo non spiegabile dalla
variabilità ordinaria (soglia 2σ superata in tutti i casi).

Questa conclusione è robusta: il MW non dipende dalla normalità, il Welch è il test
standard. L'autocorrelazione gonfia entrambi, ma la potenza dei segnali (p < 0.001
in quasi tutti i casi) rende improbabile che il rigetto sia interamente artefatto.

### 8.2 Block permutation e HAC (salto locale con split τ_price): nessun rigetto

Il permutation test con split τ_price chiede: "nel periodo immediatamente successivo
al changepoint del prezzo wholesale, il margine ha fatto un salto statisticamente
distinguibile?" La risposta è no per tutti e quattro i casi. Due ragioni principali:

**Ucraina (τ_lag = +70 giorni):** il margine ha risposto al segnale di prezzo con un
ritardo di 70 giorni. Con split a τ_price = 3 gen 2022, il "post" include sia le 10
settimane di transizione graduale sia il periodo di margine elevato a regime. Il
delta mediano post-pre risulta diluito. Il robustness check con split τ_margin (14 mar
2022) — che taglia esattamente nella rottura del margine — dà segnale significativo
(p ≈ 0.008). I due risultati sono coerenti: c'è stata una rottura reale del margine,
ma avvenuta con ritardo rispetto al prezzo.

**Iran-Israele (τ_lag benzina = −7 giorni, diesel = +28 giorni):** la finestra
pre-shock era già anomalamente elevata (δ_pre benzina = +0.086, diesel = +0.081 EUR/L).
Lo shock ha semmai *compresso* il margine rispetto al pre (δ_locale < 0). Il permutation
test con qualsiasi split non può rigettare su un delta negativo.

### 8.3 DiD: nessun segnale di specificità italiana

Nessun test DiD rigetta dopo BH. L'evidenza indica che il fenomeno dell'aumento di
margine durante le crisi è comune a tutta l'EU — coerente con il fatto che tutti i
paesi EU usano gli stessi futures wholesale (Eurobob, Gas Oil ICE) come riferimento.

### 8.4 Classificazione finale

| Evento | Carburante | Welch+MW | Perm/HAC (τ_price) | Perm/HAC (τ_margin) | DiD | Classificazione |
|---|---|---|---|---|---|---|
| Ucraina | Benzina | ✓✓ rigettano | ✗ non rigettano | ✓✓ rigettano | ✗ | **Margine anomalo positivo** |
| Ucraina | Diesel | ✓✓ rigettano | ✗ non rigettano | ✓✓ rigettano | ✗ | **Margine anomalo positivo** |
| Iran-Israele | Benzina | ✓✓ rigettano | ✗ non rigettano | ✗ non rigettano | ✗ | **Compressione margine** |
| Iran-Israele | Diesel | ✓✓ rigettano | ✗ non rigettano | ✗ non rigettano | ✗ | **Compressione margine** |
| Hormuz | Benzina | (prel.) | (prel.) | (prel.) | (prel.) | **Margine anomalo positivo** ⚠ |
| Hormuz | Diesel | (prel.) | (prel.) | (prel.) | (prel.) | **Neutro / trasmissione attesa** ⚠ |

**Nota sulla classificazione "Compressione margine" per Iran-Israele:** il margine
post-shock era sopra il 2019 (quindi "anomalo" in senso assoluto) ma era sceso rispetto
alla finestra pre-shock dello stesso evento. La classificazione descrive il movimento
*durante* l'evento. Le etichette sono descrittive dei pattern statistici, non causali.

---

## 9. Changepoint Bayesiano — Table 1 (Script 02)

Il cambio di regime nei log-prezzi è stimato con un modello piecewise-lineare
bayesiano: likelihood StudentT(ν), prior Beta(2,2) su τ, 4 catene MCMC (NUTS).
Il lag D = τ − shock_date misura l'anticipo/ritardo del changepoint rispetto allo shock.

| Evento | Serie | τ̂ | Lag D | DW | SW p | H₀ 30gg | Rhat |
|---|---|---|---|---|---|---|---|
| Ucraina | Brent | 2021-12-13 | −73 gg | 0.31 | 0.119 | NO | 1.162 ⚠ |
| Ucraina | Benzina | 2022-01-03 | −52 gg | 0.37 | 0.000 | NO | 1.001 ✓ |
| Ucraina | Diesel | 2022-01-03 | −52 gg | 0.29 | 0.000 | NO | 1.004 ✓ |
| Iran-Israele | Brent | 2025-04-28 | −46 gg | 1.20 | 0.067 | NO | 1.002 ✓ |
| Iran-Israele | Benzina | 2025-04-28 | −46 gg | 0.42 | 0.869 | NO | 1.002 ✓ |
| Iran-Israele | Diesel | 2025-05-05 | −39 gg | 0.33 | 0.489 | NO | 1.001 ✓ |
| Hormuz | Brent | 2026-02-16 | −12 gg | 1.09 | 0.150 | **SÌ** | 1.002 ✓ |
| Hormuz | Benzina | 2026-03-02 | +2 gg | 1.32 | 0.000 | **SÌ** | 1.002 ✓ |
| Hormuz | Diesel | 2026-02-23 | −5 gg | 0.93 | 0.682 | **SÌ** | 1.001 ✓ |

H₀ 30gg: rifiutata se |D| < 30 (il changepoint è troppo vicino allo shock per
escludere anticipazione di mercato).

**Nota Brent Ucraina:** Rhat = 1.162 > 1.05 indica convergenza MCMC dubbia.
Il τ̂ = 13 dic 2021 (lag = −73gg) va trattato con cautela; il risultato potrebbe
essere instabile a re-run con seed diversi.

---

## 10. Verifica Assunzione Distributiva (Script 06)

Lo script `06_distribution_check.py` confronta quattro famiglie distributive
sui residui della regressione piecewise (log-prezzi) e sui crack spread post-shock:
Normale, StudentT, Skew-Normal, Skewed-T di Fernandez-Steel. Criterio di confronto: AIC.

**Raccomandazioni per scenario:**

| Scenario | Distribuzione raccomandata |
|---|---|
| Ucraina Benzina/Diesel (log-prezzi) | **Skewed-T** |
| Ucraina Benzina/Diesel (crack spread) | **Skewed-T** |
| Ucraina Brent (log-prezzi) | StudentT (ok) |
| Iran-Israele Brent (log-prezzi) | StudentT (ok) |
| Iran-Israele Benzina (log-prezzi) | StudentT (ok) |
| Iran-Israele Diesel (log-prezzi) | Normale |
| Iran-Israele Benzina (crack spread) | Normale |
| Iran-Israele Diesel (crack spread) | **Skewed-T** |
| Hormuz Brent/Benzina (log-prezzi) | **Skewed-T** |
| Hormuz Diesel (log-prezzi) | Normale |

**Implicazioni metodologiche:**

La Skewed-T è raccomandata per la maggioranza degli scenari Ucraina e Hormuz. Questo
indica che i residui sono asimmetrici (oltre che con code pesanti): i movimenti di
prezzo non sono simmetrici attorno alla media condizionata.

Impatto sui risultati attuali:
- **I test di script 03 non sono impattati.** Welch t, MW, block perm e HAC non
  assumono una distribuzione specifica dei residui — sono robusti a questa scelta.
- **Il changepoint τ e il suo CI potrebbero spostarsi** se la likelihood fosse Skewed-T
  anziché StudentT. L'entità dello shift è valutabile solo con un re-run del modello
  bayesiano. Dove ΔAIC(Skewed-T vs StudentT) > 4, la differenza è materiale.

**Come implementare Skewed-T in PyMC (se necessario):**

```python
# Sostituire nel modello bayesian_changepoint:
# ATTUALE:
pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=y)

# CON Skew-Normal (disponibile in PyMC):
alpha_skew = pm.Normal("alpha_skew", mu=0, sigma=2)
pm.SkewNormal("obs", mu=mu, sigma=sigma, alpha=alpha_skew, observed=y)

# CON Fernandez-Steel Skewed-T (richiede likelihood custom via pm.Potential)
```

---

## 11. Analisi Annuale dei Margini

Lo script 03 produce anche un'analisi anno per anno che risponde a: il margine elevato
durante gli eventi di crisi era un fenomeno transitorio o strutturale?

**Benzina:**

| Anno | μ annua (EUR/L) | δ vs 2019 | Anomalo (2σ)? | Windfall netto M€ |
|---|---|---|---|---|
| 2019 | 0.168 | 0.000 | no | −0 |
| 2020 | 0.204 | +0.036 | no | +340 |
| 2021 | 0.182 | +0.013 | no | +126 |
| 2022 | 0.254 | **+0.085** | **Sì** | **+807** |
| 2023 | 0.235 | **+0.067** | **Sì** | +633 |
| 2024 | 0.233 | **+0.064** | **Sì** | +622 |
| 2025 | 0.255 | **+0.086** | **Sì** | **+818** |
| 2026 (parz.) | 0.274 | **+0.105** | **Sì** | +288 |

I margini sono rimasti **strutturalmente sopra il 2019** per tutti gli anni dal 2022
in poi. Il 2025 supera addirittura il 2022. Il fenomeno non è limitato ai periodi
acuti di crisi: è un livello più alto stabilizzato.

---

## 12. Limiti e Note Metodologiche

**Sulla proxy del margine:** il crack spread Eurobob/Gas Oil è una proxy del costo
wholesale ARA Rotterdam. I distributori italiani possono utilizzare contratti forward
a prezzi CIF-Genova o contratti bilaterali, introducendo una discrepanza sistematica.

**Sul dual split:** lo split τ_price risolve il problema della circolarità ma dipende
dalla stima MCMC di τ. Per Ucraina il CI 95% di τ_price copre circa 8 settimane: i
test Perm_tau e HAC_tau sono condizionati a questa stima e non vanno interpretati come
test completamente "liberi" da assunzioni modellistiche. Per Iran-Israele Diesel,
la tensione τ_price = 5 maggio vs τ_margin = 2 giugno (gap 28 giorni) è sostanziale
e giustifica la doppia implementazione.

**Sull'autocorrelazione residua:** block permutation (blocchi di 4 settimane) gestisce
parzialmente l'autocorrelazione ma non è una soluzione esatta per ρ ≈ 0.85–0.90.
Un approccio alternativo sarebbe il Newey-West con lag più lungo (es. 8–12 settimane)
o un model-based bootstrap che simuli esplicitamente il processo AR(1).

**Sulla classificazione Iran-Israele:** "Compressione margine" descrive il movimento
del margine *durante* l'evento rispetto al pre. Non esclude che il livello assoluto
del margine fosse anomalo — anzi, era fuori dalla soglia 2σ anche prima dello shock.
La distinzione tra "margine anomalo" e "compressione margine" è una distinzione temporale
(prima vs durante lo shock), non una valutazione sul livello assoluto.

**Sulla causalità:** tutte le classificazioni sono descrittive del pattern statistico.
"Margine anomalo positivo" è coerente con comportamento opportunistico ma anche con
effetti FIFO/LIFO su inventario, risk premium razionale, cost-push non catturato dalla
proxy ARA/ICE, riduzione temporanea della concorrenza. La conclusione causale
richiederebbe ulteriori evidenze (dati di inventario, margini per canale distributivo).

**Sulla distribuzione Skewed-T:** lo script 06 raccomanda Skewed-T per diversi scenari
ma la modifica al modello bayesiano non è ancora implementata nel run principale. Le
tabelle di Table 1 (τ, CI) riflettono la stima con StudentT. L'impatto qualitativo
è atteso limitato, ma andrebbe verificato con un re-run dedicato.