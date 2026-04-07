# Phase 1 Implementation Checklist

Copy this checklist locally and check off each item as you complete it. Estimated total time: **60 minutes**.

---

## Pre-Implementation (5 minutes)

- [ ] **Read** `PHASE1_ACTION_PLAN.md` (overview section)
- [ ] **Understand** what modules are available and why they exist
- [ ] **Have open:**
  - Notebook: `/Users/chloe/.../senate_llm_pipeline.ipynb`
  - Implementation guide: `PHASE1_IMPLEMENTATION_GUIDE.md`
  - Quick reference: `PHASE1_QUICK_REFERENCE.md`

---

## Step 1: Test Modules (5 minutes)

**Goal:** Verify all modules work correctly

### Tasks:
- [ ] Open terminal in project root
  ```bash
  cd /Users/chloe/LLM-Based-Personal-Profile-Extraction
  ```

- [ ] Run quick test script
  ```bash
  python3 << 'EOF'
  from modules import PATHS, NameNormalizer, extract_readable_text
  print('✓ PATHS:', PATHS['html_data'])
  print('✓ Slug:', NameNormalizer.create_slug('Bernie Sanders'))
  html = '<html><body><script>x</script>Hello</body></html>'
  print('✓ Text:', extract_readable_text(html))
  EOF
  ```

- [ ] All three checks pass? ✓ = Continue | ✗ = Debug modules

**Issues?** Check:
- [ ] `modules/` directory exists
- [ ] All 4 files present: `__init__.py`, `config.py`, `name_utils.py`, `html_processing.py`
- [ ] No syntax errors in Python files

---

## Step 2: Add Module Imports to Notebook (10 minutes)

**Goal:** Make centralized config available in notebook

### Tasks:
- [ ] In Jupyter, **create a NEW CELL** at the very top (before other cells)
- [ ] Paste this code:
  ```python
  # ============================================================================
  # IMPORT CENTRALIZED MODULES (Phase 1: Quick Wins)
  # ============================================================================
  import sys
  sys.path.insert(0, '/Users/chloe/LLM-Based-Personal-Profile-Extraction')
  
  from modules import (
      # Configuration
      PATHS,
      TASK1_DIRECT,
      TASK1_PSEUDOCODE,
      TASK1_ICL,
      PROMPT_STYLE_MAP,
      ABLATION_STYLES,
      EDUCATION_PROMPT,
      PipelineConfig,
      
      # Field definitions
      T1_FIELDS,
      T1_FIELDS_CMP,
      GT_FIELDS,
      
      # Patterns & hierarchies
      REGEX_PATTERNS,
      RELIGION_HIERARCHY,
      
      # Utilities
      NameNormalizer,
      HTMLProcessor,
      extract_readable_text,
      extract_infobox,
  )
  
  print("✓ All modules imported successfully")
  ```

- [ ] **Run this cell** (Shift+Enter)
- [ ] Check for errors: 
  - [ ] Expect: `✓ All modules imported successfully`
  - [ ] Any ImportError = Check module files

**Time check:** Should take < 5 min

---

## Step 3: Update Section 2 - Configuration (5 minutes)

**Goal:** Use centralized PATHS dictionary

Refer to: `PHASE1_IMPLEMENTATION_GUIDE.md` → Section Step 2

### Tasks:

- [ ] Find in notebook: Section 2 (Configuration) around line 70
  
- [ ] Look for this code pattern:
  ```python
  HTML_DIR   = Path("../external_data/senate_html")
  OUTPUT_DIR = Path("../outputs/senate_results")
  ```

- [ ] **Replace the entire config section** with:
  ```python
  # ════════════════════════════════════════════════════════════════════════════
  # Section 2: Configuration (Using Centralized Paths)
  # ════════════════════════════════════════════════════════════════════════════
  
  # Use centralized path configuration
  HTML_DIR   = PATHS['html_data']
  OUTPUT_DIR = PATHS['output_root']
  config_path = PATHS['groq_config']
  
  import os
  
  if not config_path.exists():
      raise ValueError("groq_config_extraction.json not found at " + str(config_path))
  
  with open(config_path) as f:
      config = json.load(f)
  
  # Extract API key
  api_key = os.getenv("GROQ_API_KEY") or config["api_key_info"]["api_keys"][0]
  
  # Extract model and provider settings
  provider = config["model_info"]["provider"]
  model = config["model_info"]["name"]
  temp = config["params"]["temperature"]
  max_tok = config["params"]["max_output_tokens"]
  
  # Initialize Groq client
  from groq import Groq
  api_client = Groq(api_key=api_key)
  
  print(f"✓ Groq API initialized from config")
  print(f"Model:       {model}")
  print(f"Provider:    {provider}")
  print(f"Temperature: {temp}  |  Max tokens: {max_tok}")
  
  html_files = sorted(HTML_DIR.glob("*.html"))
  print(f"HTML files:  {len(html_files)}")
  ```

- [ ] **Run this section** — Should see Groq API initialized ✓
- [ ] Verify no errors loading config

**Time check:** 3-5 min

---

## Step 4: Update Section 2b - Session Configuration (5 minutes)

**Goal:** Use PipelineConfig class instead of scattered variables

Refer to: `PHASE1_IMPLEMENTATION_GUIDE.md` → Section Step 4

### Tasks:

- [ ] Find Section 2b around line 114 with:
  ```python
  RUN_ALL_PROMPT_STYLES = True
  if RUN_ALL_PROMPT_STYLES:
      ACTIVE_PROMPT_STYLE = None
      STYLES_TO_RUN = ["direct", "pseudocode", "icl"]
  ```

- [ ] **Replace with:**
  ```python
  # ════════════════════════════════════════════════════════════════════════════
  # Section 2b: Session Metadata & Prompt Selection
  # ════════════════════════════════════════════════════════════════════════════
  
  # Initialize pipeline configuration
  pipeline_config = PipelineConfig()
  
  # Override any settings here if needed:
  # pipeline_config.run_all_prompt_styles = False
  # pipeline_config.active_prompt_style = "pseudocode"
  
  RUN_ALL_PROMPT_STYLES = pipeline_config.run_all_prompt_styles
  ACTIVE_PROMPT_STYLE = pipeline_config.active_prompt_style
  STYLES_TO_RUN = pipeline_config.styles_to_run
  
  # Log session metadata
  session_metadata = pipeline_config.session_metadata
  
  print("=" * 70)
  print("📋 SESSION METADATA")
  print("=" * 70)
  for key, value in session_metadata.items():
      print(f"  {key:.<20s} {value}")
  print("=" * 70)
  
  session_info = f"Running {'all 3 prompt styles' if RUN_ALL_PROMPT_STYLES else f'single style: {ACTIVE_PROMPT_STYLE}'}"
  print(f"\n✓ {session_info}\n")
  ```

- [ ] **Run this section** — Should display session metadata ✓

**Time check:** 3-5 min

---

## Step 5: Accept Consolidated Prompts (No action needed!)

**Goal:** Use imported prompts instead of re-defining them (they're already imported!)

### Tasks:
- [ ] Find Section 4 (around line 176-286) with prompt definitions
- [ ] **VERIFY** that code references `TASK1_DIRECT`, `TASK1_PSEUDOCODE`, `TASK1_ICL`
  - If they're defined locally, you can DELETE them (they're now imported)
  - [ ] Delete 110+ lines of duplicate prompt definitions
  - [ ] Just reference the imported variables

- [ ] Find where PROMPT_STYLE_MAP is defined (around line 300)
- [ ] **DELETE THIS IF PRESENT:**
  ```python
  PROMPT_STYLE_MAP = {
      "direct": TASK1_DIRECT,
      "pseudocode": TASK1_PSEUDOCODE,
      "icl": TASK1_ICL
  }
  ```
  It's already imported at the top!

**Time check:** 2-3 min (mostly deleting)

---

## Step 6: Remove Duplicate Helper Functions (10 minutes)

**Goal:** Delete 2 duplicate `create_slug()` implementations

Refer to: `PHASE1_IMPLEMENTATION_GUIDE.md` → Step 8 & 10

### Tasks:

- [ ] **FIND & DELETE Duplicate #1** (Section 8, around line 817–870)
  ```python
  def create_slug(name):
      slug = re.sub(r'\s+[A-Z]\.\s*', ' ', name).strip()
      slug = slug.replace(" ", "_")
      overrides = {"Bernard_Sanders": "Bernie_Sanders"}
      return overrides.get(slug, slug)
  ```
  Search: `def create_slug(` → Delete entire function

- [ ] **Replace usage** in Section 8 CSV building:
  - Find: `senators["wikipedia_url"] = "https://en.wikipedia.org/wiki/" + senators["name"].apply(create_slug)`
  - Replace with: `senators["wikipedia_url"] = senators["name"].apply(NameNormalizer.create_wikipedia_url)`
  
- [ ] **Find & DELETE Duplicate #2** (Section 8b, around line 1138–1210)
  ```python
  def create_wikipedia_url(senator_name):
      # ... complex implementation ...
  ```
  Search: `def create_wikipedia_url(` → Delete entire function

- [ ] **Replace usage** in Section 8b:
  - Find: `wiki_url = create_wikipedia_url(senator_name)`
  - Replace with: `wiki_url = NameNormalizer.create_wikipedia_url(senator_name)`

- [ ] **Replace in URL building:**
  ```python
  # FROM:
  senators["wikipedia_url"] = "https://en.wikipedia.org/wiki/" + senators["name"].apply(create_slug)
  senators["ballotpedia_url"] = "https://ballotpedia.org/" + senators["name"].apply(create_slug)
  
  # TO:
  senators["wikipedia_url"] = senators["name"].apply(NameNormalizer.create_wikipedia_url)
  senators["ballotpedia_url"] = senators["name"].apply(NameNormalizer.create_ballotpedia_url)
  ```

**Warning:** Make sure you replace ALL usages before deleting functions!

**Time check:** 8-10 min

---

## Step 7: Remove Duplicate Definitions (5 minutes)

**Goal:** Delete redundant field/pattern/hierarchy definitions

### Tasks:

- [ ] **Find & DELETE** duplicate `T1_FIELDS` definition (around line 641-643):
  ```python
  T1_FIELDS = ["full_name","birthdate", ...]
  ```
  ✓ Already imported at top!

- [ ] **Find & DELETE** duplicate `T1_FIELDS_CMP` definition (around line 1008)
  ✓ Already imported at top!

- [ ] **Find & DELETE** duplicate regex patterns (around line 595–613):
  ```python
  EMAIL_RE = re.compile(r"...")
  PHONE_RE = re.compile(r"...")
  # etc.
  ```
  ✓ Use `REGEX_PATTERNS` dict instead!

- [ ] **Find & DELETE** religion hierarchy definition (around line 1463–1491)
  ```python
  RELIGION_HIERARCHY = {
      "catholic": "catholic",
      # ... many lines ...
  }
  ```
  ✓ Already imported at top!

**Tip:** Search for each definition name and delete the entire block

**Time check:** 3-5 min

---

## Step 8: Test Full Pipeline (15 minutes)

**Goal:** Verify notebook still works after all changes

### Tasks:

- [ ] **Kernel → Restart** (clean slate)
- [ ] Run all cells in order (Cell → Run All)
- [ ] Check for errors:
  - [ ] Section 1: Dependencies import ✓
  - [ ] Module import cell runs ✓
  - [ ] Section 2: Configuration loads and connects to API ✓
  - [ ] Section 2b: Session metadata displays ✓
  - [ ] Section 3: HTML preprocessing works ✓
  - [ ] Section 4: Prompts available ✓
  - [ ] Section 5: Can extract from one senator (test on 1 file) ✓

- [ ] Look for any cell that produces output errors
  - [ ] Note the cell number
  - [ ] Check the ORIGINAL code vs. your edits
  - [ ] Fix and re-run

**Debugging Tips:**
- If "NameNormalizer not found" → Check module imports at top are complete
- If "PATHS dictionary error" → Check PATHS from config is imported
- If "function undefined" → You may have deleted something that's still referenced

**Time check:** 10-15 min

---

## Step 9: Commit to Git (5 minutes)

**Goal:** Save your progress with version control

### Tasks:

- [ ] Open terminal
  ```bash
  cd /Users/chloe/LLM-Based-Personal-Profile-Extraction
  ```

- [ ] Check status
  ```bash
  git status
  ```
  Should show:
  - `modules/` as new/modified
  - `experiments/senate_llm_pipeline.ipynb` as modified

- [ ] Stage changes
  ```bash
  git add modules/ experiments/senate_llm_pipeline.ipynb
  git add PHASE1_*.md NOTEBOOK_IMPROVEMENT_ANALYSIS.md
  ```

- [ ] Commit with message
  ```bash
  git commit -m "Phase 1: Consolidate configuration and utility functions

- Create modules/ package (config, name_utils, html_processing)
- Centralize paths, prompts, field definitions
- Replace duplicate name slug generation with NameNormalizer
- Consolidate HTML extraction to HTMLProcessor
- Remove ~700 lines of duplicate code
- Add comprehensive implementation documentation

Reduces notebook from 3300 to ~1900 lines (42% reduction)"
  ```

- [ ] Verify commit succeeded
  ```bash
  git log --oneline | head -5
  ```
  Should show your new commit at top

**Time check:** 3-5 min

---

## Post-Implementation (5 minutes)

### Final Checklist:

- [ ] **All cells run without errors** ✓
- [ ] **Duplicate code removed** ✓
- [ ] **Tests confirm pipeline still works** ✓
- [ ] **Changes committed to git** ✓
- [ ] **Documentation created** ✓

### Next Steps:

- [ ] **Phase 2 ready to start** (in ~1-2 weeks)
  - Consolidate evaluation loops (655 lines → 150)
  - Create metrics.py and evaluator.py
  
- [ ] **Optional:** Share progress with team
  - Show 42% code reduction
  - Demonstrate cleaner maintainability

---

## 📊 Summary Statistics

After completing Phase 1, you should have:

| Metric | Change |
|--------|--------|
| Duplicate definitions | 5+ → 1 ✓ |
| Path configurations | 10+ → 1 ✓ |
| Notebook lines | 3300 → ~1900 |
| Code reduction | ~42% ✓ |
| Implementation time | ~60 minutes ✓ |

---

## ✅ When Everything Is Complete

You'll have:
- ✓ Modular, maintainable code structure
- ✓ Centralized configuration (easy to modify)
- ✓ Reusable utility functions (usable outside notebook)
- ✓ Clear foundation for Phase 2
- ✓ Git history of your improvements

---

## 🆘 Need Help?

**If you get stuck:**

1. Check the error message carefully
2. Search in: `PHASE1_QUICK_REFERENCE.md` for similar code patterns
3. Read full instructions in: `PHASE1_IMPLEMENTATION_GUIDE.md` for that section
4. Review: `NOTEBOOK_IMPROVEMENT_ANALYSIS.md` for architectural context

**Contact points:**
- Issues with modules? Check module files for syntax
- Issues with notebook? Check import cell ran successfully
- Lost track? Return to this checklist and restart the step

---

**Ready? Start with Step 1 above!**
