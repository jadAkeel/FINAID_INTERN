# Future Research Roadmap: Forecast Select and News-Assisted Forecasting

Documented on **2026-11-20**. Status: **proposed research ideas and hypotheses; no new results or automatic authorization to run or promote them**.

This document preserves the experiments discussed with the project owner and how each could be evaluated. The aim is to improve the accuracy of the required 15–20 monthly calls if evidence supports it. More complexity, newer models, and news inputs do not guarantee improvement.

All maintained project documentation is in English. The priorities and comparisons below are research proposals, not an approved implementation plan or active job list.

## 1. Evidence to preserve before starting

| Topic | Current evidence and limits |
|---|---|
| Data | 50 anonymous indicators, X1–X50, with 316 monthly observations ending in May 2026, according to the retained data contract. Identities, units, release times, and historical vintages are undocumented. |
| Active model | Regime Adaptive selects 15–20 calls. Recorded pooled accuracy on origins 120–266 is 1,589/2,545 = 62.44%. Activation was the owner's product decision; the research promotion gate did not pass. |
| Final evaluation | Origins 268–315 were already evaluated: active accuracy 492/807 = 60.97%. They are consumed, not a fresh blind test. March–May 2026 was also inspected in an earlier study. |
| Original Chronos experiment | Chronos-2, without fine-tuning, received each indicator's historical monthly differences as an independent univariate sequence. Batch processing did not make the series exchange information. |
| Previous Chronos result | The selected blend lost 16 Validation hits and 12 Confirmation hits against the active model at matched coverage. This concerns the tested configuration, not the untested joint configuration. |
| Chronos distribution head | A small Logistic head used features of Chronos's distribution and did not improve results. This did not change Chronos's own inputs into a multivariate task. |
| TabPFN | The prepared experiment uses TabPFN-3.5, not a version named TabPFN-3.2. The last local record says download and input preparation are complete; inference awaits the user's memory-readiness message. No final accuracy is available. |
| Jev | The free route is incomplete: 273/2,304 replies in the 2026-10-10 record. There is no final result; that old record does not establish present provider availability. |
| Mistral | Mistral offers a dedicated time-series family, Timestral, as well as language models. Describing Mistral only as a language-model option was incomplete. |

Project sources: [data contract](data.md), [experiment registry](EXPERIMENT_REGISTRY.md), [corrected research outcomes](RESEARCH_OUTCOMES.md), and [final evaluation](../research/final_locked_evaluation/PROTOCOL.md). Local-only evidence not included in this documentation update: research/tabpfn35_locked/STATUS.md and research/jev_opencode_free/run_status.json.

## 2. Proposed experiment inventory

FR identifiers provide cross-references within this document; they do not represent completed or running tasks.

| ID | Experiment | Main inputs | Difference from existing evidence |
|---|---|---|---|
| FR-01 | Joint Chronos-2 forecasting | Aligned series in explicit groups | Cross-series dependencies instead of independent predictions |
| FR-02 | Chronos-2 with past covariates | Series plus historical features | Momentum, volatility, and market information inside the temporal model |
| FR-03 | Mistral Timestral | Related series and permitted covariates | A dedicated pretrained forecasting alternative not measured here |
| FR-04 | TimesFM 3 | Related series and covariates | Another pretrained reference; capabilities are version-specific |
| FR-05 | Complete TabPFN-3.5, then separate extensions | Historical feature rows and labels | Complete the prepared tabular experiment before modifying it |
| FR-06 | Bounded LightGBM / XGBoost / CatBoost study | Feature table and labels | Nonlinear relationships under the same forecasting contract |
| FR-07 | Small regularized statistical joint model | Lagged series or historical factors | A simpler reference for the value of cross-series information |
| FR-08 | iTransformer or a small temporal network | Multivariate historical windows | Learn temporal and cross-series patterns from our data |
| FR-09 | News as additional features | Timestamped news plus numerical history | Add an information source rather than only changing the algorithm |
| FR-10 | Mistral language model / Jev on fixed states | Text or a fixed numerical description | Separate information extraction from direct forecasting |
| FR-11 | Conditional fine-tuning or ensembling | Audited candidate outputs | Proceed only when evidence supports the extra complexity |

### FR-01 — Chronos-2 with cross-indicator dependencies

- **Hypothesis:** related indicators' joint history contains useful information beyond an indicator's own history.
- **Inputs:** preserve the previous differences transformation and causal cutoff, but explicitly declare shared channels and groups. Apply the frozen data-quality exclusions; do not automatically reintroduce X16.
- **Comparison:** previous independent setup versus joint setup, with identical eligible indicators, months, call counts, and conversion to direction. Do not change transformation, calibration, and blending simultaneously.
- **Grouping:** estimate relationships from available past data only and start with one declared grouping policy. An all-indicators group can be a separately declared later comparison, not an unrestricted search.
- **Input check:** inspect API tensor shapes and group identifiers to establish that information can flow within a group. Independent batching is not multivariate forecasting. Perturbing peer history is a response diagnostic, not a requirement that every prediction change.
- **Measurements:** matched-coverage hit deltas, broad-down month performance, and cost relative to independent forecasting.
- **Risks:** changing relationships, unrelated channels adding noise, and groups overfit to already inspected history.
- **Status:** proposed. See the [existing provider](../src/forecast_select/chronos_protocol.py) and [context preparation](../src/forecast_select/chronos_cache.py).

### FR-02 — Chronos-2 with historical covariates

- **Hypothesis:** a small set of past-only summaries adds information beyond jointly modeled raw histories.
- **Candidate features:** Momentum, volatility, historical breadth, and peer consensus. Freeze a small set before evaluation rather than adding all 44 features at once.
- **Arms:** independent; joint; joint plus past covariates. Compare the last arm with the joint arm to isolate the feature contribution.
- **Timing:** compute each historical feature from information available at that historical origin. Actual future Momentum or breadth is not a known-future covariate.
- **Success:** stable selected-hit improvement, not a better Brier score accompanied by worse selections.
- **Status:** proposed. The earlier Chronos distribution-head experiment does not establish that this was tested.

### FR-03 — Timestral: Mistral's forecasting model

- **Hypothesis:** a different pretrained temporal model adds value under the same task and coverage.
- **Published capabilities:** multivariate forecasting, past-only and known-future covariates, and probabilistic outputs. Announced sizes are 5M, 28M, 123M, and 209M parameters, with CPU inference. These are supplier capabilities, not project results.
- **Proposed design:** select one size after a resource preflight and before looking at evaluation outcomes; pin the checkpoint, library, and context. Compare independent and joint inputs without selecting among all sizes on final outcomes.
- **Comparators:** joint Chronos and the active model, with identical timing, eligible candidates, and coverage.
- **Outputs:** saved distributions or probabilities, monthly hits, directions, runtime, and memory.
- **Status:** proposed; not run here. [Official announcement, 2026-10-15](https://mistral.ai/news/timestral/).

### FR-04 — TimesFM 3 as another temporal reference

- **Hypothesis:** different pretraining and architecture may improve accuracy or supply complementary errors.
- **Inputs:** the same series, transformations, groups, and permitted covariates as the comparator.
- **Version boundary:** do not attribute TimesFM 3's native multivariate and covariate capabilities automatically to TimesFM 2.5 or older releases.
- **Design:** one pinned checkpoint without fine-tuning initially. Verify the selected release's license and runtime requirements before installation or use.
- **Status:** proposed. [Official repository](https://github.com/google-research/timesfm).

### FR-05 — Complete the existing TabPFN-3.5 experiment first

- **Prepared design:** 44 original numerical features plus indicator identity, the most recent 96 historical origins, 8 estimators, and a Logistic control on identical contexts.
- **First action:** finish the existing protocol only after the user reports sufficient memory. This documentation request does not revoke the recorded wait or start inference.
- **Interpretation:** replay on consumed origins 268–315 is descriptive. Do not tune an alternative on them or call them a fresh blind test.
- **Separate extension:** test context policy or add peer/news features under a new protocol. Do not change the 96-month context inside the existing run and mix the results.
- **Method:** historical X and y condition predictions for new rows. In the configured usage, fit prepares the context; it is not neural-weight fine-tuning.
- **Status:** prepared but incomplete according to the last local record. Existing local protocol and runner: research/tabpfn35_locked/PROTOCOL.md and research/tabpfn35_locked/run.py. They are not included in this documentation update. [Prior Labs](https://github.com/PriorLabs/TabPFN).

### FR-06 — Tabular models trained on project data

- **Candidates:** LightGBM and XGBoost. CatBoost already has retained results and must not be described as an untested family.
- **Hypothesis:** trees capture interactions between Momentum, volatility, and market conditions more effectively than Logistic on this dataset.
- **Comparison:** identical features, targets, eligible indicators, and selections. Begin with one candidate and a small declared search, without adding news to the initial comparison.
- **Revisiting CatBoost:** requires an isolated difference, such as new information or matched coverage. Another search on the same known outcomes is not independent evidence.
- **Risk:** monthly observations are limited, and indicators within a month are dependent. Row count is not an independent-sample count.
- **Status:** proposed. [LightGBM](https://lightgbm.readthedocs.io/en/stable/Features.html), [XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/model.html), and [historical comparisons](../research/reference_models/README.md).

### FR-07 — A limited statistical joint model

- **Hypothesis:** a simple model of lagged relationships captures some multivariate benefit with less complexity.
- **Candidates:** regularized/Bayesian VAR, or a model of a few historical factors rather than all 50 series directly.
- **Design:** few lags, audited transformations and quality, and regularization or dimensionality reduction. Estimate factors and relationships inside the available history at each origin.
- **Comparison:** independent versus joint statistical reference, followed by comparison with foundation models on the same direction/selection task.
- **Risk:** an unrestricted 50-series VAR with many lags introduces many parameters relative to the number of months.
- **Status:** proposed; choose the regularization and implementation before execution. [VAR reference](https://www.statsmodels.org/stable/vector_ar).

### FR-08 — A temporal network trained on our data

- **Initial candidate:** a small iTransformer. Another family needs a distinct protocol and hypothesis, not an open-ended architecture competition.
- **Hypothesis:** multivariate windows encode temporal relationships not captured by manually engineered features.
- **Inputs:** time-by-indicator arrays with missingness masks and frozen transformations. Define the project's monthly target explicitly instead of silently adopting a library's default target.
- **Comparison:** an appropriately simpler model with the same information and coverage. Search, early stopping, and normalization remain inside development.
- **Priority:** later, because the monthly history is short and training from scratch may overfit.
- **Status:** proposed. [Original iTransformer implementation](https://github.com/thuml/iTransformer).

## 3. FR-09 — Connecting news to numerical forecasts

### 3.1 Hypothesis and missing information

News available before the forecast decision may add information about market reversals or group movements beyond numerical history. Positive sentiment is not assumed to imply Up, and news usefulness is not presumed.

First obtain a documented dictionary linking each X to its identity, units, market, source, release schedule, and definition changes. Do not infer that X1 is oil or X44 is bonds from correlations alone, or promote descriptive group labels into verified identities.

Without that dictionary, a limited market-wide news study is possible. Indicator-specific linking remains conditional on a verified mapping.

### 3.2 Sources and availability evidence

Start with one or two specified sources, such as official central-bank statements, economic releases, or issuer publications after identifying the relevant series. Check historical coverage, archive quality, permitted use, and cost before choosing a news or consensus-data provider. This document does not authorize subscriptions or external data transmission.

Retain:

| Proposed field | Purpose |
|---|---|
| article_id and source_url | Trace the source |
| source and language | Record provenance and language |
| first_published_at_utc | Establish the first verifiable publication time |
| version_published_at_utc and content_hash | Identify the actual text version and prevent later revisions entering earlier forecasts |
| first_seen_at_utc or historical_archive_evidence | Establish when the method could access the content; collecting it today does not establish historical availability |
| retrieved_at_utc | Record current collection time separately from publication |
| event_time | Keep event time separate from announcement time |
| source_snapshot and dedup_cluster | Recover permitted text and avoid counting syndicated duplicates repeatedly |
| entity/group mapping and mapping_evidence | Record a supported relationship to a series or group |
| extractor_model/prompt/schema hashes | Freeze the extraction method |

An article first published before cutoff is insufficient if the available text includes updates made afterward. Retrospective reporting that explains the target month's completed outcome cannot serve as a prior input.

### 3.3 Preserve the information clock

At origin t, numerical inputs remain available through t−1 and training labels through t−2 under the existing contract. Freeze the snapshot time and time zone before evaluation, and require the specific news version to have been published and accessible by then.

Example matching the saved forecast: May 2026 origin predicts the May-to-June move, with numerical features through April. In the first conservative comparison, news also stops at a predefined April snapshot. May news is not allowed merely because the workbook contains May values.

A scenario using news newer than the last numerical observation needs a separately declared availability contract, decision time, and evaluation. Do not silently shorten the lag or use the target month's outcomes.

Unknown future news values must not enter Chronos/Timestral/TimesFM as known-future covariates. An announcement's scheduled date may be known; its actual decision or release value is not. Economic surprise additionally requires a consensus estimate timestamped before the announcement. Without that archive, do not claim a measured surprise.

### 3.4 Convert news into testable features

Proposed flow:

    Versioned, timestamped news
        → availability checks and deduplication
        → verified market/group/indicator mapping
        → extraction of a few event fields
        → past-only temporal aggregation
        → join with numerical features
        → save forecast before outcome
        → compare with numbers alone

Freeze a small subset of candidate features:

- Unique-story counts and coverage intensity, distinguishing weak source coverage from an absence of events.
- Event proportions for monetary tightening/easing, inflation pressure, slowdown, commodity supply disruption, or geopolitical risk. Do not impose the same effect direction on every indicator.
- Reported direction and severity where supported by the text; otherwise unknown.
- Narrative disagreement or uncertainty under a fixed computational definition.
- A verified group's exposure to a specified event type.

A single aggregation window and roughly 6–8 features are a proposed starting point, not accepted settings. No articles, retrieval failure, neutral reporting, and extraction uncertainty are different states; do not fill all with zero.

### 3.5 Role of a Mistral language model or Jev

Use the language model as an extractor: read the supplied text and return fixed fields with article_id and a short supporting excerpt or text location. Allow unknown. It must not invent prices, indicator identities, or events absent from the supplied material.

Pin one model/version, prompt, and category definition before examining forecast hits. Manually audit an extraction sample without exposing targets, and cache the outputs rather than regenerating them until accuracy improves. Low temperature alone does not guarantee identical outputs.

Mistral language models can be evaluated for extraction; Timestral is the separate numerical forecasting candidate. Jev is conditional on provider availability and a protocol. Completing its old numerical questions is not a news experiment. [Mistral prompting capabilities](https://docs.mistral.ai/inference/prompting).

Even with valid historical text, pretrained model weights may contain knowledge of later events. An instruction to use only the supplied text reduces risk but cannot prove the absence of future knowledge. Historical results remain exploratory; stronger evidence requires timestamped predictions on genuinely new events using a fixed model.

### 3.6 News comparisons that isolate the contribution

| Arm | Change | Question |
|---|---|---|
| N0 | Numerical model without news | Fixed reference on the same months and coverage |
| N1 | Same model plus deterministic story/coverage counts, without an LLM | Does news activity alone add value? |
| N2 | Same model plus rule-based event features | Does event type add value beyond activity? |
| N3 | Same model plus fixed LLM extraction | Does text interpretation add value beyond simpler alternatives? |
| N4 | After evidence of signal, add the same features to a second model | Is the gain due to information or the estimator? |
| N5 | Separate news-based broad-down/breadth study | Does better market timing translate into selected-hit gains? |

Do not run every model, arm, and weight simultaneously. Start with one numerical reference and a small feature set; expand only when the incremental comparison warrants it. Dependent work uses reviewed predecessor outputs, not merely a completed job.

For tabular models, join news features to indicator-month rows. For temporal models, supply aligned past-covariate channels and missingness masks according to the API. A market-timing task predicts breadth or broad-down events and is evaluated separately before introducing direction flips.

Conditional examples include a rate announcement linked to a verified fixed-income group, or an oil-supply disruption linked to a verified energy series. Learn and test the effect rather than fixing a positive/negative sign from the event name.

## 4. FR-10 — Direct language-model forecasting

This is lower priority than measuring additional documented information. A fixed numerical/textual state can be evaluated with a Mistral language model or Jev, provided:

- The numerical comparator sees the same information; understanding terms does not establish forecasting skill.
- Report every declared arm, retain raw responses and prompt versions, and do not search prompts on final outcomes.
- Resume existing Jev jobs and caches rather than duplicating them; no permission for new external transmissions is inferred.
- Changing inputs, questions, or model creates a new experiment rather than a continuation with mixed results.
- The [existing Jev protocol](../research/jev_zero_shot_locked/PROTOCOL.md) remains authoritative. Additional local evidence: research/jev_opencode_free/README.md, not included in this documentation update.

## 5. FR-11 — Conditional fine-tuning and ensembling

**Fine-tuning:** establish a pretrained-use reference first. Consider adaptation only with a specific rationale, adequate data, and an acceptable resource budget. Compare before/after on identical tasks, and report training and inference costs separately. A supplier's adaptation service is not automatically local or free; verify the actual route before implementation.

**Ensembling:** first measure complementary errors outside model selection. Does the candidate recover the reference's mistakes more often than it spoils correct calls? Diversity alone is insufficient. Fit blend weights on earlier OOF predictions only, freeze them, and retain the best individual model as a comparator.

Existing Chronos blends, confidence corrections, and directional mixes have negative results. Revisiting them requires a materially different hypothesis, information source, or evaluation design. Consult the [negative-results registry](SELECTION_GROUP_FAILED_REGISTRY.md).

## 6. Common evaluation protocol to freeze before execution

### Information and target

1. The primary target is the t-to-t+1 direction under the project definition. Sudden drops, breadth, and continuous values are separate secondary targets.
2. Every arm uses the same cutoff and availability lag. Do not randomly shuffle temporal train/test splits.
3. Distinguish coefficient fitting, hyperparameter selection, pretrained-context preparation, and neural-weight fine-tuning; record what actually happens.
4. Fit scalers, imputation, grouping, PCA, feature/relationship selection, and calibration on permitted past data only.
5. Previously held-out outcomes may become eligible history for later forecasts. This does not make them a fresh holdout again or permit using the current forecast target.

### Fair comparisons and measurements

- Use the same eligible indicators, quality exclusions, and months. Primary coverage matches the active model's monthly k within 15–20. Lower coverage is a separately declared comparison.
- Report hits/calls, paired monthly hit differences, accuracy, Up/Down counts, and direction-specific correctness. Record the entering and displaced call for each replacement.
- Separate directional AUC on all candidates, correctness AUC within selected calls, and top-k accuracy. Do not substitute one metric to declare success.
- Evaluate Brier and calibration when probabilities are claimed; ranking scores are not verified individual correctness probabilities.
- Diagnose regimes, years, and broad-down months without selecting a rule after seeing the best slice.
- Use paired moving-block bootstrap by month; a proposed starting point is 6-month blocks and 5,000 replicates, with sensitivity reported. Indicator rows within a month are not independent samples.
- Record memory, runtime, cost, and missing coverage. Partially completed months are not a final result for the full experiment.

### Decision rule

Proposed gate, to approve and freeze before outcomes: positive hit delta at matched coverage; an accuracy gain above a prespecified practical threshold; a positive lower bound for the paired comparison; sufficient stability outside selection; and no clear contradiction in the later period. Do not presume a required 65% result or promise a 3–4-point gain.

Origins 120–266 have been reused extensively. Replays are exploratory. Origins 268–315 and March–May are consumed as well. They may describe behavior under a frozen protocol, but independent adoption evidence requires forecasts recorded on months not used for selection.

The retained dataset ends in May 2026; no later evaluation is claimed. Declare a future evaluation period, duration, and stopping rule beforehand, and save predictions before outcomes. General benchmark leadership is not proof of success on this task.

Operational data updates also need consistent provenance: the [current forecast code](../src/forecast_select/future_regime_forecast.py) rejects workbook hashes that disagree with frozen research results. Do not promise that appending rows alone is sufficient or bypass that check; plan the update path separately when implementing.

## 7. Proposed experiment artifacts

Create these only during an authorized execution; the names do not imply that files already exist:

    research/<experiment_id>/PROTOCOL.md
    research/<experiment_id>/manifest.json
    research/<experiment_id>/predictions.parquet
    research/<experiment_id>/monthly_metrics.csv
    research/<experiment_id>/summary.json
    research/<experiment_id>/REPORT.md

The manifest records revision and uncommitted diff; data/news/feature/code/checkpoint hashes; package versions; cutoffs; fitting/selection/evaluation windows; groups; parameters and seeds; device; commands; runtime and memory; and completed months. News studies additionally retain snapshots, identity evidence, extractor versions, and prompts.

Before interpreting results, verify future invariance by changing data/news after cutoff and checking earlier inputs/predictions. Check target alignment, coverage equivalence, calibration, and replacement accounting.

Retain rejections and ties. Update the [experiment registry](EXPERIMENT_REGISTRY.md) after an actual run. A protocol or library installation alone does not establish completion or accuracy.

## 8. Proposed order and dependencies

1. **Establish the reference:** freeze snapshots, information rules, and comparators; review existing failures.
2. **Existing-number branch:** complete TabPFN after the user's readiness message; evaluate FR-01 and then FR-02 on accepted predecessor inputs. Timestral and TimesFM are later alternatives, not mandatory simultaneous jobs.
3. **News branch:** establish identity, source, and availability evidence; begin with N0/N1 before more complex extraction. General market news can be studied before every identity is known; indicator-specific links require verified mapping.
4. **Combined decision:** review accuracy and cost together; pursue fine-tuning, ensembling, or a heavier network only when evidence supports it.
5. **Forward verification:** freeze the candidate and record genuinely new forecasts before any adoption claim.

This is a revisable research order, not an executed DAG. No task store, workers, or scheduled jobs were created.

## 9. Decisions needed before implementation

- First experiment, checkpoint, and runtime/memory/cost limits.
- Verified indicator identities and a usable news archive.
- Minimum practical gain, forward period, duration, and stopping rule before outcomes.
- Text extractor, local/API route, and material allowed to leave the machine.
- Data-update and provenance workflow with reviewed production changes.

This documentation does not authorize purchases, paid APIs, uploads, or production changes. Future executors own cleanup of temporary tool sessions and processes and restoration of the disabled baseline. Do not start persistent helpers or unnecessary WSL instances.

## 10. Preparation and verification record

- Prepared on 2026-11-20 from the discussion, local files, and primary references linked by model family. Converted to English at the owner's request.
- Reference revision: 98e765425be8ed304315a276e25f901500d30509, with pre-existing uncommitted work; this revision does not establish a clean working tree.
- Supplier references establish available capabilities, not project performance. Pin versions, licenses, and dependencies again when implementing.
- Documentation only: no model fitting, inference, news-archive collection, or production/data changes were performed.
- The original documentation pass verified UTF-8, 26 local links across its three files, the 11 experiment IDs, and git diff --check. Language-conversion checks are recorded separately below; these are not new model tests.
- Before the original documentation pass, 143 protected files under src, configs, tests, data, artifacts/active, and reports had aggregate SHA-256 428f9dce78dd89cbe5e43d0fabdc1aaccc23f63629c74e8b51b705ecd6012fe3; that pass preserved the digest.

### English-language conversion checks

The owner requested English project files and a GitHub update on 2026-11-20.
The roadmap was renamed to FUTURE_RESEARCH_ROADMAP.md and its incoming links
were updated. Historical report/instruction filenames were retained for
compatibility, with their content translated into English.

- A content scan passed for 437 text files and 11 ZIP/Office containers,
  including ignored poster-build sources outside model caches. No Arabic
  text matched the scan.
- Extracted text from 12 PDF files, covering 15 pages, contained no Arabic
  matches. This was a text audit, not an OCR audit of embedded images.
- All numerical tokens were preserved in order across 627 translated lines
  in six research, archive, note, and poster-instruction documents.
- Historical archive reports retain their figures and superseded claims,
  with explicit quarantine notices rather than a renewed promotion claim.
- All 21 local links in the roadmap, README, and experiment registry resolved.
- All 14 poster-kit manifest entries matched locally and inside the updated
  ZIP. The local poster generator's AST was unchanged except for its
  translated instruction string.
- The 143 protected model/data/configuration/test/report files retained the
  aggregate digest above. No forecasting experiments or model tests were run.
- git diff --check passed. The documentation update excludes pre-existing
  uncommitted research implementations and their generated model artifacts.
