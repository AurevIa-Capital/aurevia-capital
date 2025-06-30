generate a json file in this format, for the first 10 watches for each brand (refer to screenshot). The links for each brand is provided.
[
  {
    "brand": "Patek Philippe",
    "model_name": "Nautilus 5711/1A",
    "url": "https://watchcharts.com/watch_model/22871-patek-philippe-nautilus-5711-stainless-steel-5711-1a/overview",
    "source": "generated"
  },
  {
    "brand": "Patek Philippe",
    "model_name": "Aquanaut 5167A",
    "url": "https://watchcharts.com/watch_model/22557-patek-philippe-aquanaut-5167-stainless-steel-5167a/overview",
    "source": "generated"
  },
]

Top Tier
Patek Philippe
Rolex
Audemars Piguet
Vacheron Constantin
https://watchcharts.com/watches/brand/patek+philippe
https://watchcharts.com/watches/brand/rolex
https://watchcharts.com/watches/brand/audemars+piguet
https://watchcharts.com/watches/brand/vacheron+constantin


Mid Tier
Omega
Tudor
Hublot
https://watchcharts.com/watches/brand/omega
https://watchcharts.com/watches/brand/tudor
https://watchcharts.com/watches/brand/hublot

Designer / Entry-Level Collector
Tissot
Longines
Seiko



  Implementation Priority

  Phase 1 (High Impact, Low Risk):
  1. Factory Pattern for Models - eliminates model creation duplication
  2. Builder Pattern for Configuration - centralizes config logic
  3. Centralized Data Store - reduces file system coupling

  Phase 2 (Medium Impact, Medium Risk):
  1. Strategy Pattern for Pipelines - improves modularity
  2. Observer Pattern for Events - adds monitoring
  3. Command Pattern for CLI - improves maintainability

  Expected Benefits

  - Code Reduction: From ~40% to ~10% duplication
  - Improved Testability: Easy mocking with dependency injection
  - Better Maintainability: Single responsibility principle
  - Enhanced Monitoring: Event-driven pipeline visibility

  Would you like me to implement any of these specific patterns, starting with the highest-impact, lowest-risk improvements?
