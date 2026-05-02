La struttura del documento (in lingua italiana) deve comprendere: abstract, Introduzione, Metodologie/Dati, Risultati, Discussione Finale e Bibliografia. 
I link citati, ponili come voci di bibliografia.


Lunghezza testo 	Massimo 3000 parole
Tabelle e immagini: Massimo 5



# Abstract: ---
# Introduzione: ---
	- in questi anni soggetti a forti aumentiu dei carburanti, in particolare ultimamente con la chiusura di stretto di hormuz. Le gente si lamenta che le compagnie petrolifere ci lucrino sopra, aumentando irragionamvolmente i prezzi alla pompa, creando disagi a mlioni di lavoratori. Ma sara' veramente cosi? 
	- La nostra H0 ipotesi nulla: in prossimità temporale di shock geopolitici che coinvolgono Paesi fornitori di petrolio greggio o semilavorati verso l’Italia, i distributori italiani di carburante non generano profitti anomali, con aumenti dei prezzi che risultano coerenti con la crescita dei costi di approvvigionamento e delle materie prime.
	- Ipotesi alternativa (H₁): in prossimità temporale di shock geopolitici che coinvolgono Paesi fornitori di petrolio greggio o semilavorati verso l’Italia, i distributori italiani di carburante generano profitti anomali, applicando aumenti di prezzo superiori a quelli giustificabili dalla sola crescita dei costi di approvvigionamento e delle materie prime.


	- pro
	- vogliamo analizzare 

# Dati e Metodologie:
## Dati
Raccoglie da fonti pubbliche tre serie di dati settimanali:
	- MIMIT (ex MISE) — Open Data Carburanti
		https://opendatacarburanti.mise.gov.it/categorized
		prezzi giornalieri, lordi (prezzo finale alla pompa), al litro, per ogni impianto italiano, per benzina e diesel, self service mode

	- SISEN/MASE — Prezzi settimanali con componente fiscale
		https://sisen.mase.gov.it/dgsaie/
		https://sisen.mase.gov.it/dgsaie/api/v1/weekly-prices/report/export?type=ALL&format=CSV&lang=it
		tasse e accise settimanali

-> si calcola il prezzo netto giornaliero Euro/L: prezzo_pompa − tax_wedge, sia per benzina che diesel

	- Futures wholesale europei — Eurobob B7H1 Futures (benzina) (2017-01-02 → 29/04/2026) e Gas Oil ICE Futures (diesel) 2015-01-01 → 28/04/2026), che
		rappresentano il costo all'ingrosso che i distributori effettivamente pagano sul
		mercato spot di Amsterdam-Rotterdam-Anversa.

		Fonte: Eurobob_B7H1_date.csv  ( derivante da scraping su TradingView NYMEX:B7H1!)
		https://it.tradingview.com/symbols/NYMEX-B7H1!/
        I futures ICE sono quotati in USD/tonnellata metrica (USD/ton).
		Per convertire in USD/litro usiamo la densità liquida del prodotto,
		derivata dalla composizione molecolare media (approccio molare).
		I prezzi wholesale, quotati originariamente in USD per tonnellata ($"USD/t"$), sono stati convertiti in Euro al litro ($"EUR/L"$) per garantire la coerenza dimensionale con i prezzi alla pompa. La procedura segue due step:
			1. *Conversione fisica*: Divisione per la densità specifica del carburante:
			- *Benzina*: $0.74 "kg/L" arrow.r 1351 "L/t"$
			- *Diesel*: $0.84 "kg/L" arrow.r 1190 "L/t"$
			2. *Conversione valutaria*: Utilizzo del tasso di cambio spot settimanale EUR/USD. (yfinance (EURUSD=X), https://finance.yahoo.com/quote/EURUSD=X)


-> margine = prezzo_netto (ex-tasse, da SISEN) − prezzo_wholesale (futures €/L)
	margine_gasolio = gasolio_net  −  Gas Oil Futures (€/L)
	margine_benzina = benzina_net  −  Eurobob B7H1 Futures (€/L)


## Metodologie:



# Risultati
	- Prima osservazione generale sulla composizione del prezzo finale di diesel e benzina: accise e IVA componente molto prominente
		-> wholesale/02_decomposizione_gasolio.png
		-> wholesale/02_benzina_gasolio.png
	
	-





	- Prezzo del Brent in euro/barile — scaricato da Yahoo Finance (ticker BZ=F) [https://finance.yahoo.com/quote/BZ%3DF/], convertito da USD a EUR usando il tasso di cambio della stessa settimana. [https://finance.yahoo.com/quote/EURUSD=X/]
	- Prezzi alla pompa italiani senza tasse e accise — dall'EU Weekly Oil Bulletin, il bollettino
	ufficiale europeo che ogni lunedì pubblica i prezzi al consumo netti di accise e IVA per
	tutti i paesi EU [https://energy.ec.europa.eu/data-and-analysis/weekly-oil-bulletin_en]. In particolare usiamo il prospetto "Price developments 2005 onwards"
	
