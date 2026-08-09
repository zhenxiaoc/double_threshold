/****************************************
Household effect sizes - Figures 2 & S2
Dofile #1 in sequence
This do-file standardizes all outcome variables into effect sizes for use
in Figures 2 and S2 in Science paper publication.
Household-level variables
Stata version: 12.1
*****************************************/

clear all
set maxvar 15000
cap log close
version 12.1

if "${master_test}!"!="1" {
	cap do "1_set_globals.do"
	if _rc==601 {
		di "Please run the 1_set_globals do-file before continuing"
	}
}

use "${dta_working}\pooled_hh.dta", clear
cap log close

/* This first local stores each of the table names. Each table name then
consists of a series of outcome variables that belong in the table. This 
makes it possible to simultaneously reference both the table and the variable
names within that table. */
loc tables "assets increv consumption food_security finance"
loc assets "asset_index asset_tot_value asset_prod_index asset_prod_value asset_hh_index asset_hh_value"
loc increv "ranimals_month iagri_month ibusiness_month ipaidlabor_month percep_econ"
loc consumption "ctotal_pcmonth cfood_pcmonth cnonfood_pcmonth cdurable_pcmonth"
loc food_security "index_foodsecurity fs_enoughfood fs_adultskip fs_wholeday fs_childskip fs_twomeals"
loc finance "loan_totalamt loan_informalamt loan_formalamt sav_totalamt sav_depositamt"
loc outcomevars `assets' `increv' `consumption' `food_security' `finance'

/* Control vars includes all the variables apart from geographic cluster on
which the randomization was stratified; this will allow for all controls to
be called into every variable. */
loc controlvars control_*


/* Standardizing all outcome variables
Variables are standardized to control mean and standard deviation for 
the given round */
foreach var in `outcomevars' {
	foreach t in bsl end fup {
		forvalues i = 1/6 {
			qui sum `var'_`t' if treatment==0 & country==`i'
			if `r(N)' > 0 {
				replace `var'_`t' = (`var'_`t' - `r(mean)')/`r(sd)' if country==`i'
			}
		}
	}
}		

* Creating empty matrices where the with family-grouped standard effects will be stored for graphs
loc matrix_count: word count `outcomevars'
foreach t in end fup {
	matrix hh_`t' = J(`matrix_count', 6, .)
}


/*
**********************************
2. ADJUSTING GEOGRAPHIC CLUSTER CODES, 
DEALING WITH MISSING VARIABLES FROM
BASELINE, AND INCLUDING SHORT SURVEY
CONTROLS
**********************************
*/

/* Ensuring geographic cluster dummies are different between
countries */
replace geo_cluster = geo_cluster + (country * 1000) 

/* Adjusting control variables for each country, based on what
variables are missing. */

/* 1. Generating controlvar_m dummy for countries where control var is nonmissing
	but hh has missing value. replacing controlvar with 0 as well. */
levelsof country, loc(countries)
foreach cvar of varlist control_* {
	cap drop `cvar'_m
	gen `cvar'_m = 0
	foreach country in `countries' {
		qui summ `cvar' if country==`country'
		if `r(N)' != 0 {
			replace `cvar'_m = 1 if `cvar'==. & country==`country'
			replace `cvar' = 0 if `cvar'_m ==1
			}
		}
	loc dcontrols `dcontrols' `cvar'_m
}

/* 2. Replacing the control variable equal to zero if 
	it doesn't exist for the country in question(ie, is 
	missing for all observations) */
foreach country in `countries' {
	foreach controlvar of varlist `controlvars' {
		di "`controlvar'"
		cap assert missing(`controlvar') if country==`country'
		if _rc==0 {
			replace `controlvar' = 0 if country==`country' // replacing with 0 if whole country is missing
		}
	}
}

/* Generating a dummy for if the country doesn't have the variable
at baseline for ALL observations (distinct from it being missing
for SOME observations). */
foreach outcomevar in `outcomevars' {
	cap drop m_country_`outcomevar'_bsl
	gen m_country_`outcomevar'_bsl = 0
	foreach country in `countries' {
		/* Asserting that all values are missing for the country.
		If true (!_rc), it will recode the variable for all obs
		in the country to 1 */
		cap assert missing(`outcomevar'_bsl) if country==`country'
		if !_rc {
			di "`country' is missing `outcomevar' at baseline"
			replace m_country_`outcomevar'_bsl = 1 if country==`country'
		}
	}
}

/************************************************************
3. Running regressions, summary statistics, exporting results
************************************************************/
log using "$log\regressions_hh", replace

*Pooled Country Results
loc tablecount: word count `tables'

loc count_end = 1
loc count_fup = 1

forvalues i = 1/`tablecount' {
	loc first 1
	local table: word `i' of `tables'
	global table_name "`table'_table.txt"
	foreach t in end fup {
		foreach var in ``table'' {
			loc ss 0
			** Including short survey controls for questions that were asked at short survey round only (only applies to EL1)
			if inlist("`var'","ctotal_pcmonth", "cfood_pcmonth", "cnonfood_pcmonth", "cdurable_pcmonth", "loan_totalamt", "loan_informalamt", ///
			"loan_formalamt") ///
			| inlist("`var'", "sav_totalamt", "ranimals_month", "iagri_month", "ibusiness_month", "ipaidlabor_month") {
				if "`t'"=="end" {
					loc ss 1
				}
			}
			cap confirm variable `var'_bsl
			* Specification when variable exists at baseline
			if !_rc {
				if "`t'"=="end" {
					cap drop m_`var'_bsl
					generate m_`var'_bsl = missing(`var'_bsl)
					replace m_`var'_bsl = 0 if m_country_`var'_bsl		
					replace `var'_bsl = 0 if missing(`var'_bsl)
				}
				qui sum `var'_`t' if treatment==0
				loc control_mean `r(mean)'
				loc control_sd `r(sd)'
				qui sum `var'_bsl if m_`var'_bsl!=1
				loc baseline_mean `r(mean)'
				areg `var'_`t' treatment `var'_bsl m_`var'_bsl m_country_`var'_bsl ///
					control_* `=cond(`ss'==1, "css_p? css_g? css_h?", "")', absorb(geo_cluster) ///
					cluster(rand_unit)
				matrix V = e(V)
				matrix b = e(b)
				loc itt = _b[treatment]
				loc sd = sqrt(V[1,1])
				loc tstat = b[1,1]/sqrt(V[1,1])
				loc pvalue = 2*ttail(`e(df_r)', abs(`tstat'))
				loc df = `e(df_r)'
				loc first 0
			}
			* Specification when variable does not exist at baseline
			else if _rc == 111 {

				qui sum `var'_`t' if treatment==0
				loc control_mean `r(mean)'
				loc control_sd `r(sd)'
				loc baseline_mean
				areg `var'_`t' treatment `controlvars' ///
					control_* `=cond(`ss'==1, "css_p? css_g? css_h?", "")', absorb(geo_cluster) ///
					cluster(rand_unit)
				matrix b = e(b)
				matrix V = e(V)
				loc itt = _b[treatment]
				loc sd = sqrt(V[1,1])
				loc tstat = b[1,1]/sqrt(V[1,1])
				loc pvalue = 2*ttail(`e(df_r)', abs(`tstat'))
				loc df = `e(df_r)'
				loc first 0
			}
			mat hh_`t'[`count_`t'', 1]=`control_mean' // storing control mean
			mat hh_`t'[`count_`t'', 2]=`control_sd' // storing control standard deviation
			mat hh_`t'[`count_`t'', 3]=`itt' // storing itt coefficient
			mat hh_`t'[`count_`t'', 4]=`sd' // storing itt se
			mat hh_`t'[`count_`t'', 5]=`pvalue' // storing p-value
			mat hh_`t'[`count_`t'', 6]=`df' // storing degrees of freedom
			loc ++count_`t'
		}
	}
}

matrix list hh_end
matrix list hh_fup

//save "${dta_working}\standardized_hh_postanalysis.dta", replace

** Creating Dataset used in Figures 2 and S2
gen varname = ""
forvalues i = 1/`count_end' {
	loc value: word `i' of `outcomevars'
	replace varname = "`value'" if _n==`i'
}

foreach t in end fup {
	svmat hh_`t'
	rename hh_`t'1 controlmean_`t'
	rename hh_`t'2 controlsd_`t'
	rename hh_`t'3 B_Treatment_`t'
	rename hh_`t'4 SE_Treatment_`t'
	rename hh_`t'5 p_value_Treatment_`t'
	rename hh_`t'6 DF_`t'
}

keep varname controlmean* controlsd_* B_Treatment* SE_Treatment* p_value_Treatment* DF_*
keep if !mi(controlmean_end)
save "${dta_working}\outcome_matrix_hh.dta", replace

cap log close


