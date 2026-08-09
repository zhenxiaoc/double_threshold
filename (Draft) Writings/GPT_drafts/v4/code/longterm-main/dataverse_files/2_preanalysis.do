**************************************
** Preanalysis Do-File
** TUP Science Paper
** Stata version: 12.1
**************************************

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

foreach dataset in hh mb {
	use "${dta}\pooled_`dataset'", clear

	/*** 1. Creating block stratification dummies and cluster variable (for standard errors)
	geo = village number /!\ this is the original variable for each country, not used in analysis
	
	geo_cluster = block stratification variable
		India, Ethiopia, Pakistan: village number (same as geo) for all households
		Honduras, Peru, Ghana: village number for treatment and control households in treatment villages
		Honduras, Peru, Ghana: 0 for all control households in control villages (because this group did not receive stratification)
	
	rand_unit = unit of randomization (for standard error clustering)
		Standard errors are clusterd at the unit of randomization.
		India, Ethiopia, Pakistan SE's are clustered at household level (cluster size of 1 for household outcomes, clustered at household-level for individual outcomes)
		Honduras, Peru, Ghana had two-stage randomization
			1. control villages: all households were control households (randomization at village level)
			2. within treatment villages, households were randomized into treatment or control groups (household level randomization)
		Treatment and control households within treatment villages: SE's are clustered at household level (cluster size of 1 for household outcomes, clustered at household-level for individual outcomes)
		Control households in control villages (where only village-level randomization took place): SE's are clustered at the village level for both adult and household outcomes
	
	Please refer to SOM text 1 for a more detailed explanation of block stratification controls and standard error clustering
	*/
	label variable geo "Original geographic cluster variable (village), not used in analysis"	
	gen geo_cluster = geo /* All households for whom entry into the program was ultimately determined at the HH level 
	(either in a simple randomization or a 2-stage randomization) 
	have a control in regressions for their block strata */
	
	replace geo_cluster = 0 if spillover_treat==0 /* However, households chosen into control communities did not have randomization determined at the individual level, and therefore should not 
	have a block strata control.*/
	label variable geo_cluster "Modified geographic cluster variable, block stratification controls"

	if "`dataset'"=="hh" {
		loc id id
	}
	else if "`dataset'"=="mb" {
		loc id id_hh
	}
	encode `id', gen(rand_unit)
	qui sum rand_unit
	replace rand_unit = geo + `r(max)' if spillover_treat==0 // randomization unit at the village level for control households in control villages (Peru, Honduras, Ghana only)
	
	label variable rand_unit "Unit of randomization, for clustering standard errors" /* This code ensures that all individuals for whom randomization 
	was determined at the individual level are not clustered (ie cluster size of one), but those in pure control villages share a "randomization unit" */ 
	

	
	*** 2. Creating country-specific treatment interactions for equality of treatment test
	forvalues i = 1/6 {
		gen treat_country`i' = (treatment==1 & country==`i') if !mi(treatment)
	}
	tab country, gen(control_dcountry)

	/* 3. Generating controlvar_m dummy for countries where control var is nonmissing
		but hh has missing value. replacing controlvar with 0 as well. */
	local controlvars control_*
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
		
	/* 4. Replacing the control variable equal to zero if 
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
	loc controlvars `controlvars' `dcontrols' //renaming controlvars local to include the missing dummies
	
	** 5. Putting outcome variables in local
	if "`dataset'"=="hh" {
		loc assets "asset_index asset_tot_value asset_prod_index asset_prod_value asset_hh_index asset_hh_value"
		loc increv "ranimals_month iagri_month ibusiness_month ipaidlabor_month percep_econ"
		loc consumption "ctotal_pcmonth cfood_pcmonth cnonfood_pcmonth cdurable_pcmonth"
		loc food_security "fs_enoughfood fs_adultskip fs_wholeday fs_childskip fs_twomeals"
		loc finance "loan_totalamt loan_informalamt loan_formalamt sav_totalamt sav_depositamt"
		loc outcomevars `assets' `increv' `consumption' `food_security' `finance'
	}
	else if "`dataset'"=="mb" {
		loc health "ddays_nomiss dailyliving_score percep_health percep_life index_stress dnoworry"
		loc time_use "time_work time_agri time_animals time_business time_paidlabor"
		loc empowerment "pol_vote pol_party pol_leadermeet pol_leaderask dec_food_major dec_educ_major dec_health_major dec_himprov_major dec_hmgmt_major"
		loc outcomevars `health' `time_use' `empowerment'
	}
	
	/* 6. Generating a dummy for if the country doesn't have the variable
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
	/* 7. Dummy for missing outcome variable at baseline (different from when missing for whole country) */
	foreach var in `outcomevars' {
		cap drop m_`var'_bsl
		generate m_`var'_bsl = missing(`var'_bsl)
		replace m_`var'_bsl = 0 if m_country_`var'_bsl		
		replace `var'_bsl = 0 if missing(`var'_bsl)
	}
	
	/* 8. Incorporating short survey control variables: 
		Replacing missings with zeros */
	qui summ country
	loc obs `r(N)'
	foreach var of varlist css_* {
		replace `var'=0 if missing(`var')
		qui summ `var'
		loc count `r(N)'
		display "obs = `obs' and css_var = `count'"
		}

	save "${dta_working}\pooled_`dataset'", replace
}

