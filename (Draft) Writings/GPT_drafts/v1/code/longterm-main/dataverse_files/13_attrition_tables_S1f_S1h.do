****************************************************************
** Attrition Tables
** TUP Science Paper
** Stata version: 12.1
****************************************************************

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

tempfile temp_hh temp_ad temp_merge

use "${dta_working}\\index_hh_vars.dta", clear

*** Setting up locals
loc hhvars index_ctotal_bsl index_foodsecurity_bsl asset_index_bsl ind_fin_bsl ind_increv_bsl
loc advars index_time_bsl index_health_bsl index_mental_bsl index_political_bsl index_women_bsl
loc baseline_vars index_ctotal_bsl index_foodsecurity_bsl asset_index_bsl ind_fin_bsl index_time_bsl ind_increv_bsl index_health_bsl index_mental_bsl index_political_bsl index_women_bsl
loc countries

foreach var in `hhvars' {
	replace `var'=. if m_`var'==1
}
save `temp_hh'

use "${dta_working}\\index_mb_vars.dta", clear

foreach var in `advars' {
	replace `var' = . if m_`var'==1
}

* Collapse member-level data to household level so as to be able to include mb and hh level variables in single regression
collapse `advars', by(id_hh)
rename id_hh id

merge 1:1 id using `temp_hh', gen(merge_ind)

cap log close
log using "${log}\\attrition_hh", replace


** Replacing Missing values with 0s and generating missing variable dummies
levelsof country, loc(countries)
foreach var in `baseline_vars' {
	cap drop `var'_m
	gen `var'_m = 0
	forvalues i = 1/6 {
		qui sum `var' if country==`i'
		replace `var'_m = 1 if missing(`var') & country==`i'
		replace `var' = 0 if `var'_m ==1
	}
	loc m_vars `m_vars' `var'_m
}

** Generating interaction terms
foreach var in `baseline_vars' {
	cap drop tX`var'
	gen tX`var' = `var'*treatment
	loc interactions `interactions' tX`var'
	}	
foreach var in `m_vars' {
	cap drop tX`var'
	gen tX`var' = `var'*treatment
	loc minteractions `minteractions' tX`var'
	}		
	

loc country Ethiopia Ghana Honduras India Pakistan Peru
loc count: word count `country'	

** Column 1: Attrition on Treatment
loc first 1

* Generating surveyed var
foreach t in end fup {
	gen surveyed_`t' = 1 - attrition_`t'
}

* Panel A. Pooled
foreach t in end fup {
	qui summ surveyed_`t'
	loc outcomemean= r(mean)
	areg surveyed_`t' treatment, absorb(geo_cluster)
	est store Pooled1
	outreg2 [Pooled1] using "${tables}\\attrition\attrition_pooled_1", ///
	`=cond(`first'==1, "replace", "")' auto(2) title("Column 1 Panel A") ///
	ctitle(`t' Pooled) addstat("Outcome mean", `outcomemean') excel
	loc first 0
}

* Panel B. By Country
loc first 0
foreach t in end fup {
	forvalues i=1/6 {
		loc countryname: word `i' of `country'
		display "Country is `countryname'"
		qui summ surveyed_`t' if country==`i'
		if r(N)!=0 {
			loc outcomemean= r(mean)
			areg surveyed_`t' treatment if country==`i', absorb(geo_cluster)
			est store `countryname'1
			outreg2 [`countryname'1] using "${tables}\\attrition\attrition_pooled_1", ///
			`=cond(`first'==1, "replace", "")' auto(2) title("Column 1 Panel B") ///
			ctitle(`t' `countryname') addstat("Outcome mean", `outcomemean') excel
		}
	}
}

** Column 2: Attrition on Baseline Vars
loc first 1

* Panel A. Pooled
foreach t in end fup {
	qui summ surveyed_`t'
	loc outcomemean= r(mean)
	areg surveyed_`t' treatment `baseline_vars' `m_vars', absorb(geo_cluster)
	est store Pooled2
	outreg2 [Pooled2] using "${tables}\\attrition\attrition_pooled_2", ///
	`=cond(`first'==1, "replace", "")' auto(2) title("Column 2 Panel A") ///
	ctitle(`t' Pooled) addstat("Outcome mean", `outcomemean') excel
	loc first 0
	}

* Panel B. By Country
loc first 0
foreach t in end fup {
	forvalues i=1/6 {
		loc countryname: word `i' of `country'
		display "Country is `countryname'"
		qui summ surveyed_`t' if country==`i'
		if r(N)!=0 {
			loc outcomemean=r(mean)
			areg surveyed_`t' treatment `baseline_vars' `m_vars' if country==`i', absorb(geo_cluster)
			est store `countryname'2
			outreg2 [`countryname'2] using "${tables}\\attrition\attrition_pooled_2", ///
			`=cond(`first'==1, "replace", "")' auto(2) title("Column 2 Panel B") ///
			ctitle(`t' `countryname') addstat("Outcome mean", `outcomemean') excel
		}
	}
}

** Column 3: Attrition on Baseline Vars & Interactions
loc first 1
* Panel A. Pooled
foreach t in end fup {
	qui summ surveyed_`t'
	loc outcomemean=r(mean)
	areg surveyed_`t' treatment `baseline_vars' `m_vars' `interactions' `minteractions', absorb(geo_cluster)
	est store Pooled3
	test treatment `interactions'
	loc pval=r(p)
	outreg2 [Pooled3] using "${tables}\\attrition\attrition_pooled_3", ///
	`=cond(`first'==1, "replace", "")' auto(2) title("Column 3 Panel A") ///
	ctitle(`t' Pooled) addstat("Outcome mean", `outcomemean', "P-value", `pval') excel
	loc first 0
}

* Panel B. By Country
loc first 0
foreach t in end fup {
	forvalues i=1/6 {
		loc countryname: word `i' of `country'
		display "Country is `countryname'"
		qui summ surveyed_`t' if country==`i'
		if r(N)!=0 {
			loc outcomemean=r(mean)
			areg surveyed_`t' treatment `baseline_vars' `m_vars' `interactions' `minteractions' if country==`i', absorb(geo_cluster)
			est store `countryname'3
			preserve
			keep if country==`i'
			test treatment `interactions'
			loc pval=r(p)
			restore
			outreg2 [`countryname'3] using "${tables}\\attrition\attrition_pooled_3", ///
			`=cond(`first'==1, "replace", "")' auto(2) title("Column 3 Panel B") ///
			ctitle(`t' `countryname') addstat("Outcome mean", `outcomemean', "P-value", `pval') excel
		}
	} 
}	

cap drop tX*
cap log close

exit
