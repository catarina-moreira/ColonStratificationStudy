---
name: sato-tubular-straightening
description: Analyzes, explains, sets up, and runs the SATO Python project for straightening 3D tubular objects from volumetric medical images. Use when the user wants to understand the SATO paper-to-code mapping, inspect the GitHub repository, prepare inputs, execute the straightening scripts, troubleshoot environment or data issues, or assess reproducibility. Do not use for unrelated Python projects, generic medical image segmentation training, or claims of full paper reproduction when only the core straightening code is available.
---

# SATO Tubular Straightening

## Purpose
Use this skill to help the agent understand and operate the SATO project for straightening 3D tubular anatomical structures from 3D medical images.

Repository: [Yanfeng-Zhou/SATO](https://github.com/Yanfeng-Zhou/SATO)

This skill is for:
- paper-to-code interpretation
- repository inspection
- environment setup
- input validation
- script execution
- debugging
- reproducibility assessment

This skill is not for:
- training new segmentation models from scratch
- benchmarking unrelated medical imaging methods
- claiming complete reproduction of all paper experiments if the repository only provides the core straightening implementation
- clinical validation or diagnostic claims

## When to Use This Skill
Use this skill when the user asks to:
- explain what the SATO project does
- connect the paper to the GitHub implementation
- run the SATO scripts
- prepare image, segmentation, or centerline inputs
- debug script failures
- evaluate whether the repo is reproducible
- adapt the method to airway, vessel, or other tubular imaging workflows

## Required Capabilities
The agent must be able to:
- read Python code
- inspect repository structure
- interpret scientific computing dependencies
- understand volumetric medical image processing
- reason about centerlines, local frames, and cross-sectional resampling
- detect repo limitations and missing pipeline components
- explain risks caused by poor upstream data quality

## Input Contract

### Minimum Inputs
The agent should expect:
- a local clone or accessible copy of the SATO repository
- a 3D image volume
- a precomputed centerline file
- the user’s goal, such as:
  - explain the method
  - run the code
  - debug an error
  - assess reproducibility

### Common Optional Inputs
The agent may also receive:
- a segmentation mask
- the associated paper PDF
- screenshots or traceback logs
- desired output folder
- imaging modality context such as CT, MRI, micro-CT, bronchoscopy-related airway volumes, or vascular imaging

### Input Assumptions to Validate
The agent must validate:
- whether the centerline already exists
- whether the centerline format matches what the code expects
- whether image coordinates and centerline coordinates are aligned
- whether voxel spacing and axis ordering are known
- whether segmentation masks are binary or label-valued as expected

## Output Contract
The agent should produce one or more of the following:
- a concise explanation of the SATO method
- a code-level walkthrough of the implementation
- setup commands for a Python environment
- exact commands to run the scripts
- a checklist of required inputs
- a diagnosis of execution failures
- a reproducibility assessment with explicit caveats
- a summary of what the repository implements versus what the paper claims end-to-end

The agent must clearly separate:
- what is implemented in the repository
- what is only described in the paper
- what must be supplied externally by the user

## Core Knowledge the Agent Should Apply

### Scientific Domain
The agent should understand:
- 3D medical image volumes
- segmentation masks
- centerlines
- voxel spacing
- interpolation and resampling
- curved planar reformation concepts
- tubular anatomy such as vessels and airways

### Geometric Domain
The agent should understand:
- tangent vectors along discrete curves
- local frames for sweeping cross-sections along a centerline
- limitations of Frenet frames
- recursive frame transport or swept-frame logic
- practical sampling of perpendicular slices in 3D

### Software Domain
The agent should understand:
- Python CLI execution
- `numpy`
- `scipy`
- `scikit-image`
- `SimpleITK`
- possible use of multiprocessing or `tqdm_pathos`
- medical image file handling through Python scientific libraries

## Procedure

### Step 1: Inspect the Repository
Read the repository metadata and identify:
- the main scripts
- dependency declarations
- expected input files
- whether sample data is included
- whether there is complete end-to-end documentation

Prioritize:
- `README.md`
- `requirements.txt`
- the main straightening scripts
- any helper utilities

### Step 2: Determine Scope
Classify the user’s request into one of these modes:
1. explanation only
2. setup and execution
3. debugging
4. reproducibility audit
5. adaptation guidance

If the user’s intent spans multiple modes, complete them in that order:
1. explanation
2. setup
3. execution
4. debugging
5. reproducibility assessment

### Step 3: Identify the Actual Implemented Pipeline
Determine which parts are present in code:
- image straightening
- segmentation straightening
- batch or parallel execution
- postprocessing

Determine which parts are not present or not fully packaged:
- segmentation model training
- centerline extraction
- branch handling beyond what is explicitly coded
- downstream clinical or analytic tasks claimed in the paper

State these boundaries explicitly.

### Step 4: Validate Inputs Before Running
Before giving execution commands, verify:
- image file path exists
- centerline file path exists
- centerline shape and datatype are plausible
- centerline coordinates appear to be in the correct reference frame
- output directory is writable
- the chosen script matches the task:
  - image volume straightening
  - segmentation straightening
  - parallel batch processing

If the user has not provided input data, explain what they need to supply.

### Step 5: Prepare the Environment
Generate setup instructions using the repo’s actual dependency list.

At minimum, guide the user to:
- create a clean Python environment
- install dependencies from `requirements.txt`
- verify importability of the main packages
- note any OS-specific issues if visible from the code or package requirements

If exact versions are missing or underspecified, state that clearly.

### Step 6: Map Paper Concepts to Code
Explain how the repository operationalizes the method:
- how the centerline drives resampling
- how local orientation is updated along the path
- how cross-sections are sampled
- how image and segmentation workflows differ
- where heuristic cleanup steps are applied

If the code contains brittle logic, hard-coded values, or ambiguous assumptions, highlight them.

### Step 7: Produce Execution Commands
Provide exact commands the user can run.

The commands must:
- use the actual script names from the repository
- include placeholder paths that are easy to replace
- distinguish between image straightening and segmentation straightening
- avoid inventing unsupported flags or arguments

If the command cannot be fully constructed from the repo alone, say what is missing.

### Step 8: Troubleshoot Failures
If execution fails, classify the failure into one of these buckets:
- environment failure
- missing dependency
- file path error
- malformed centerline input
- image-centerline coordinate mismatch
- unexpected array shape
- interpolation or sampling failure
- postprocessing or mask logic issue

For each failure:
1. explain the likely root cause
2. explain how to verify it
3. propose the smallest safe fix
4. avoid claiming success until the input assumptions are validated

### Step 9: Assess Reproducibility
If asked about reproducibility, evaluate:
- whether the repository is enough to reproduce the core straightening method
- whether the repository is enough to reproduce all reported paper experiments
- what external preprocessing steps are missing
- whether documentation is sufficient for a third party

State the result in one of these forms:
- core method reproducible with external preprocessing
- partially reproducible
- not fully reproducible from official repo alone

### Step 10: Communicate Limits Clearly
Always disclose:
- dependence on upstream segmentation quality
- dependence on accurate centerline extraction
- possible coordinate-system pitfalls
- gaps between the paper narrative and the released code
- that straightening quality may be evaluated indirectly if the repo lacks direct metrics

## Failure Handling

### Failure Mode: Missing Centerline
Action:
- explain that SATO expects a precomputed centerline
- do not imply the repository necessarily extracts it
- ask for or describe the required centerline format if inferable from code

### Failure Mode: Shape or Format Mismatch
Action:
- inspect expected `.npy` or array structure
- compare it to the provided input
- suggest a conversion only if the expected schema is clear from the repository

### Failure Mode: Coordinate Misalignment
Action:
- warn that even valid files can fail if image and centerline coordinate systems differ
- recommend checking spacing, orientation, and axis order
- avoid guessing orientation corrections without evidence

### Failure Mode: Dependency Problems
Action:
- identify the missing package
- point to installation via `pip`
- note if the package is platform-sensitive or version-sensitive

### Failure Mode: Weak Documentation
Action:
- reconstruct the likely workflow from code
- label inferred steps as inferred
- do not present inferred usage as official documentation

### Failure Mode: Paper-Repo Mismatch
Action:
- explain that the repository may implement only the geometric core
- separate the paper’s full experimental claims from the released code artifacts

## Quality Bar
The agent should aim to be:
- precise
- conservative
- reproducibility-aware
- explicit about assumptions
- honest about missing components

The agent must not:
- fabricate missing scripts
- invent sample data
- claim clinical readiness
- overstate reproducibility
- assume centerline semantics without checking the code

## Recommended Response Pattern
When responding, prefer this structure:
1. what the project does
2. what the repo actually contains
3. what inputs are required
4. how to run it
5. what can go wrong
6. what is and is not reproducible

## Example Triggers

### Positive Triggers
- Explain the SATO paper and how the GitHub code implements it
- Help me run the SATO repository on my airway CT data
- What inputs do I need to use SATO for 3D tubular straightening
- Debug this SATO script error
- Is the SATO repository enough to reproduce the paper

### Negative Triggers
Do not trigger this skill for:
- Train a new nnU-Net model on airway segmentation
- Compare generic Python packages for image registration
- Build a web app for radiology image upload
- Summarize an unrelated AI paper
- Perform clinical diagnosis from CT scans

## Minimal Execution Template
When the user wants to run the project, produce:

1. environment setup steps
2. list of required inputs
3. exact command template
4. output expectations
5. troubleshooting checklist

Example template:

- Environment:
  - create virtual environment
  - install `requirements.txt`
- Inputs:
  - image volume path
  - centerline path
  - optional segmentation path
- Run:
  - execute the relevant straightening script with user paths
- Validate:
  - confirm output file exists
  - inspect output dimensions
  - inspect anatomical plausibility
- Troubleshoot:
  - check coordinate conventions
  - check centerline schema
  - check dependency imports

## Reproducibility Summary Template
If asked whether the project is reproducible, answer in this pattern:

- **Core algorithm**: implemented / partially implemented / unclear
- **Paper preprocessing pipeline**: present / missing / partial
- **End-to-end experiments**: reproducible / partially reproducible / not reproducible from repo alone
- **Main blocker**: centerline extraction, missing data, sparse documentation, missing evaluation scripts, or another explicit cause

## Notes
This skill should be kept lean. If supporting references are added later, place them in:
- `references/` for input schemas, dependency notes, and geometry notes
- `assets/` for command templates or expected file-layout templates
- `scripts/` only for deterministic helper utilities


