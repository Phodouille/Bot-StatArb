# StatArb Bot — CLAUDE.md (orchestrateur)

Ce fichier est chargé par Claude Code (session principale + tous les sous-agents).
Il contient les règles globales et les contrats d'interface entre modules.
**Ne jamais le contourner ou le réécrire sans validation humaine explicite.**

## Contexte projet

Bot de statistical arbitrage : PCA + mean-reversion sur 40-50 actions tech US
liquides du S&P 500. Deux features différenciantes vs un stat arb classique :
1. Fenêtre PCA glissante **adaptative** (s'ajuste à la volatilité du marché,
   pas une fenêtre fixe)
2. Filtres macro (VIX + spreads de crédit) qui réduisent/suspendent le trading
   en période de stress, quand le mean-reversion casse

4 niveaux de benchmark = **un seul codebase**, flags on/off dans la config :
buy-and-hold S&P500 → equal-weight sectoriel → stat arb PCA classique
(fenêtre fixe, sans filtres) → notre modèle (adaptatif + filtres macro).

Broker : Alpaca, paper trading, **compte cash** (pas de règle PDT, mais
attention au délai de règlement T+2 → utiliser `buying_power`, jamais `cash`,
pour calculer le capital disponible). Passage en live = un seul changement
d'endpoint Alpaca, ne rien coder qui suppose le paper trading en dur.

## Stack

Python, pandas (pas polars), scikit-learn (PCA), Streamlit (dashboard),
Alpaca API (alpaca-py), pytest.

## Structure du repo

```
data/        ingestion, nettoyage, stockage des prix          (agent: data-ingestion)
signals/     PCA, z-score, génération de signaux              (agent: signals)
backtest/    moteur de backtest, 4 niveaux de benchmark        (agent: backtest)
execution/   passage d'ordres Alpaca, exécution paper/live     (humain — pas d'agent dédié pour l'instant)
risk/        position sizing, drawdown guard, kill switch      (agent: risk)
dashboard/   app Streamlit                                     (full-stack)
db/          fichiers de données (parquet/sqlite)               -
tests/       tests unitaires + intégration                      (gouvernance)
docker/      déploiement                                        (full-stack)
notebooks/   exploration EDA uniquement, jamais de logique prod  -
```

---

## CONTRAT 1 — Signal contract (FIGÉ, ne pas modifier sans accord des deux côtés)

Format de sortie produit par `signals/`, consommé par `execution/` :

```python
signals = {
    "AAPL": {"direction": "long", "z_score": -2.3, "weight": 0.05},
    "MSFT": {"direction": "short", "z_score": 2.1, "weight": 0.04},
}
```

- `direction` : `"long"` / `"short"` / absent du dict si pas de position voulue
  sur ce ticker (ne pas mettre `"flat"`, juste omettre la clé)
- `z_score` : float, signé, valeur du z-score au moment du calcul
- `weight` : float, poids cible en fraction du portefeuille (positif, le signe
  est porté par `direction`)
- Le dict ne contient QUE les tickers avec une position désirée (pas tout
  l'univers à chaque fois)

## CONTRAT 2 — Data contract (PROPOSITION — à valider en équipe avant de lancer les agents)

Interface produite par `data/`, consommée par `signals/`, `backtest/`, `risk/`.

**Stockage** (`db/prices.parquet` par défaut) — format long :

| date (datetime64) | ticker (str) | open | high | low | close | volume |

**Fonction d'accès exposée par le module data** (signature à respecter) :

```python
def load_returns(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Retourne une matrice wide : index=date, columns=ticker, valeurs=log-return.
    - Jamais de NaN dans le DataFrame retourné.
    - Ticker avec >5% de trous sur la fenêtre demandée → exclu (logué).
    - Trous ponctuels (<=2 jours) → forward-fill.
    - INTERDIT : retourner une date > la date `end` demandée (no look-ahead).
    """
```

Si l'équipe préfère un autre format (ex: wide stocké directement, ou SQLite
au lieu de parquet pour l'auditabilité côté gouvernance), modifier cette
section AVANT de lancer les agents en parallèle — c'est le seul changement
qui casse tout le reste si fait après coup.

---

## Règles globales (s'appliquent à tous les modules et tous les agents)

1. **No look-ahead bias** : un calcul à la date J ne doit JAMAIS utiliser une
   donnée datée > J. Vrai pour la PCA, les z-scores, les filtres macro, le
   backtest. C'est le bug le plus probable et le plus difficile à détecter.
2. **Paramètres de risk jamais en dur dans le code** : tout passe par un seul
   fichier de config (`config/risk_params.yaml` ou équivalent — à créer),
   pas de constante éparpillée dans plusieurs fichiers.
3. **Univers d'actions** : liste figée dans `config/universe.py` (ou
   équivalent) — **TODO : AAPL, MSFT, GOOGL, AMZN, META**
   une fois arrêtée, pour que tous les agents travaillent sur la même liste.
4. **Conventions de code** : type hints obligatoires sur les fonctions
   publiques, docstrings Google style, un test pytest minimum par fonction
   publique dans `tests/`.
5. **Credentials** : clés Alpaca uniquement via `.env` (jamais commitées,
   jamais en dur dans un prompt ou un fichier). Voir `.env.example`.
6. **Coûts de transaction** : à modéliser explicitement dans le backtest
   (5-10 bps/trade), le turnover est ce qui tue l'alpha en stat arb.

## Risk guardrails (valeurs de référence — module `risk/`)

- Position max par titre : 5-10% du portefeuille
- Max drawdown : si > 5% → coupure automatique du bot (kill switch)
- Exposition nette quasi market-neutral (long ≈ short en valeur)
- Kill switch manuel exposé sur le dashboard
- Capital paper : 100k$ fictifs sur Alpaca

## Points de vigilance (à rappeler à chaque agent si pertinent)

- Survivorship bias sur l'univers backtesté → à mentionner comme limite dans
  le rapport, pas à "corriger"
- Geler les paramètres de calibration à partir de la semaine 8 — ne pas
  réoptimiser indéfiniment
- Si le Sharpe du backtest dépasse 3 → chercher le bug, ne pas célébrer
- Garder un plan B pour la soutenance (captures/vidéos si la démo live plante)
