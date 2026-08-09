/* 
Variable-by-variable analysis dofile (Tables S2 and S5 - household level outcomes)
** Stata version: 12.1

This dofile runs through the outcome variables from the household-level dataset, 
storing each set of variables into a separate table in the same folder. It
requires that all country codes are assigned, and that each country's intervention
has been broken into a binary treatment/control variable named "treatment".
*/

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

/*
**********************************
1. SETTING LOCALS
**********************************
*/
/* This first local stores each of the table names. Each table name then
consists of a series of outcome variables that belong in the table. This 
makes it possible to simultaneously reference both the table and the variable
names within that table. */

use "${dta_working}\pooled_hh.dta", clear
cap log close

loc tables "consumption food_security assets finance increv"
loc assets "asset_index asset_tot_value asset_prod_index asset_prod_value asset_hh_index asset_hh_value"
loc increv "ranimals_month iagri_month ibusiness_month ipaidlabor_month percep_econ"
loc consumption "ctotal_pcmonth cfood_pcmonth cnonfood_pcmonth cdurable_pcmonth"
loc food_security "fs_enoughfood fs_adultskip fs_wholeday fs_childskip fs_twomeals"
loc finance "loan_totalamt loan_informalamt loan_formalamt sav_totalamt sav_depositamt"
loc outcomevars `assets' `increv' `consumption' `food_security' `finance'

/* Checking that each outcome variable exists for each country at each round 
baseline will have the largest amount of variation */
foreach var in `outcomevars' {
	foreach t in bsl end fup {
		di "`var' at `t'"
		tab country if !missing(`var'_`t')
	}
}

/**********************************
2. GETTING SUMMARY STATS, RUNNING REGRESSIONS,
EXPORTING THE RESULTS
**********************************/
log using "$log\regressions_hh", replace

* Panel A: Pooled Country Results
loc tablecount: word count `tables'

loc count_end = 1
loc count_fup = 1

forvalues i = 1/`tablecount' {
	loc first 1
	local table: word `i' of `tables'
	global table_name "`table'_table.xls"
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
			// Specification when baseline value of outcome variable exists
			if !_rc {
				qui sum `var'_`t'
				loc obs `r(N)'
				qui sum `var'_`t' if treatment==0
				loc control_mean `r(mean)'
				loc control_sd `r(sd)'
				qui sum `var'_bsl if m_`var'_bsl!=1
				loc baseline_mean `r(mean)'
				// Regression storing only F-test of equality of treatment by country
				areg `var'_`t' treat_country? `var'_bsl m_`var'_bsl m_country_`var'_bsl ///
					control_* `=cond(`ss'==1, "css_p? css_g? css_h?", "")', absorb(geo_cluster) cluster(rand_unit)
				test treat_country1==treat_country2==treat_country3==treat_country4==treat_country5==treat_country6
				loc equal_treat_F = `r(F)'
				loc equal_treat_p = `r(p)'
				// Regression that results are drawn from (pooled treatment, not broken down by country)
				areg `var'_`t' treatment `var'_bsl m_`var'_bsl m_country_`var'_bsl ///
					control_* `=cond(`ss'==1, "css_p? css_g? css_h?", "")', absorb(geo_cluster) cluster(rand_unit)
				outreg2 treatment using "${tables}\\variable_components\household\pooled\\${table_name}", ///
					`=cond(`first'==1, "replace", "")' addstat("Control mean", ///
					`control_mean', "Baseline mean", `baseline_mean', ///
					"F-test equality of treatment", `equal_treat_F', "p-value equality of treatment", ///
					`equal_treat_p') auto(2)
				loc first 0
				
			}
			// Specification when baseline value of outcome variable does not exist
			else if _rc == 111 {
				qui qui sum `var'_`t'
				loc obs `r(N)'
				qui sum `var'_`t' if treatment==0
				loc control_mean `r(mean)'
				loc control_sd `r(sd)'
				loc baseline_mean
				// Regression storing only F-test of equality of treatment by country
				areg `var'_`t' treat_country? control_* `=cond(`ss'==1, "css_p? css_g? css_h?", "")', absorb(geo_cluster) cluster(rand_unit)
				test treat_country1==treat_country2==treat_country3==treat_country4==treat_country5==treat_country6
				loc equal_treat = `r(F)'
				loc equal_treat_p = `r(p)'
				// Regression that results are drawn from (pooled treatment, not broken down by country)
				areg `var'_`t' treatment control_* ///
					`controlvars' `=cond(`ss'==1, "css_p? css_g? css_h?", "")', absorb(geo_cluster) cluster(rand_unit)
				outreg2 treatment using "${tables}\\variable_components\household\pooled\\${table_name}", ///
					`=cond(`first'==1, "replace", "")' addstat("Control mean", ///
					`control_mean', "Baseline mean", `baseline_mean' ///
					"F-test equality of treatment", `equal_treat_F', "p-value equality of treatment", ///
					`equal_treat_p')) auto(2)
				loc first 0
			}
		}
	}
}

*Panel B: Individual Country Results
loc country_tables "Ethiopia Ghana Honduras India Pakistan Peru"
loc countrycount: word count `country_tables'
forvalues j = 1/`countrycount' {
	forvalues i = 1/`tablecount' {
		loc first 1
		loc table: word `i' of `tables'
		loc ctry: word `j' of `country_tables'
		global table_name "`ctry'_`table'.xls"
		foreach t in end fup {
			foreach var in ``table'' {
				// If round is "end" (endline 1), include the short survey controls
				loc ss 0
				** Including short survey controls for questions that were asked at short survey round only (only applies to EL1)
				if inlist("`var'","ctotal_pcmonth", "cfood_pcmonth", "cnonfood_pcmonth", "cdurable_pcmonth", "loan_totalamt", "loan_informalamt", ///
				"loan_formalamt") ///
				| inlist("`var'", "sav_totalamt", "ranimals_month", "iagri_month", "ibusiness_month", "ipaidlabor_month") {
					if "`t'"=="end" {
						loc ss 1
					}
				}
				qui summ `var'_`t' if country == `j'
				//Confirming existence of outcome variable by country
				if `r(N)' !=0 {
					cap confirm variable `var'_bsl
					// Specification with baseline controls
					if !_rc {
						qui sum `var'_`t' if country == `j' & treatment == 0
						loc control_mean `r(mean)'
						qui sum `var'_bsl if country==`j' & m_`var'_bsl!=1
						loc baseline_mean `r(mean)'
						areg `var'_`t' treatment `var'_bsl m_`var'_bsl m_country_`var'_bsl ///
							control_* `=cond(`ss'==1, "css_p? css_g? css_h?", "")' if country ///
							== `j', absorb(geo_cluster) cluster(rand_unit)
						outreg2 treatment using "${tables}\\variable_components\household\country\\${table_name}", ///
							`=cond(`first'==1, "replace", "")' addstat("Control mean", ///
							`control_mean', "Baseline mean", `baseline_mean') auto(2)
						loc first 0
					
					}
					// Specification without baseline controls
					else if _rc == 111 {
						qui qui sum `var'_`t'
						loc obs `r(N)'
						qui sum `var'_`t' if country == `j' & treatment==0
						loc control_mean `r(mean)'
						loc baseline_mean

						areg `var'_`t' treatment control_* `=cond(`ss'==1, "css_p? css_g? css_h?", "")' ///
							if country == `j', absorb(geo_cluster) cluster(rand_unit)
						outreg2 treatment using "${tables}\\variable_components\household\country\\${table_name}", ///
							`=cond(`first'==1, "replace", "")' addstat("Control mean", ///
							`control_mean', "Baseline mean", `baseline_mean') auto(2)
						loc first 0
					}
				}
			}
		}
	}
}

save "${dta_working}\pooled_hh_postanalysis.dta", replace
cap log close
exit
