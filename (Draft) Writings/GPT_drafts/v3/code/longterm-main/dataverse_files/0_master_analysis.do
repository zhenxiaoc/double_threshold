**************************************
** Master Do-File
** TUP Science Paper
** Stata version: 12.1
** Updated: 10/31/2017
**************************************
clear all
set maxvar 15000
cap log close
set more off
version 12.1

** SET DIRECTORY HERE
/* Have your path include the folder structure up through the Folder Name 
"ScienceDataRelease", or whatever you have called the top level folder where
you have downloaded the data. */
global path "YOUR PATH"
global ssc "$path\ado files"

** CALL USER-WRITTEN COMMANDS
sysdir set PLUS "${ssc}"

** 1. Setting globals for folder structure
// this step is required before running any other dofiles in the sequence
cap do "$path\\dofiles\1_set_globals.do"

if _rc==601 {
	cap do "1_set_globals.do"
	if _rc==601 {
		di "Please change the global 'path' before continuing"
		stop
	}
}

** 2. Pre-analysis dofile - Creates block stratification variables, country weights
do "$do\2_preanalysis.do"

** 3. Tables 3 and S4: Indexed Family Outcomes Analysis
/* Produces tables for pooled and country-level indexed family outcomes
This dofile also creates datasets (member and household level) with indexed outcomes.
These datasets are used to create effect size graphs for country-by-country sample 
for indexed family outcomes */
do "$do\3_index_analysis_tables_3&S4.do"

** 4. Figures 2, 3 and S2, S3: Effect Size Graphs
// Creates effect size datasets for household level variables
do "$do\4_effectsizes_hh_figs2&S2_1.do"
// Creates effect size dataset for individual level variables
do "$do\5_effectsizes_mb_figs2&S2_2.do"
// Creates effect size graphs for Pooled sample
do "$do\6_effectsizes_figs2&S2_3.do"
// Creates effect size graphs for Country-by-Country sample
do "$do\7_ctry_effectsizes_figs3&S3.do"

** 5. Table 5: Quantile Analysis, Pooled Sample
do "$do\8_quantile_analysis_table5.do"

** 6. Tables S2 and S5: Variable Components, pooled and country-level
do "$do\9_vars_hh_tables_S2&S5.do" // household
do "$do\10_vars_mb_tables_S2&S5.do" // individual */

** 7. Tables S1a-S1e: Baseline summary statistics
do "$do\11_orthogonality_tables_S1a.do"
do "$do\12_orthogonality_tables_S1b_S1e.do"

** 8. Tables S1f-S1h: Attrition
do "$do\13_attrition_tables_S1f_S1h.do"

** 9. Table S3: Manski-Lee Bounded Attrition
do "$do\14_bounded_attrition_table_S3.do"

** 10. Tables S6: Spillover analysis (Ghana, Honduras, Peru)
do "$do\15_spillover_tables_S6.do"

** 11. Table 4: Calculates updated results
do "$do\16_nondurable_consumption_analysis_table_4.do"
