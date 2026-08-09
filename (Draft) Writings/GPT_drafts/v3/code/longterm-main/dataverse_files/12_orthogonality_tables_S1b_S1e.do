*****************************************************************
** Summary Statistics & Balance Tables
** TUP Science Paper
** Stata version: 12.1
*****************************************************************

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

tempfile temp_hh temp_adult temp_merged

loc hhvars index_ctotal_bsl index_foodsecurity_bsl asset_index_bsl ind_fin_bsl ind_increv_bsl
loc advars index_time_bsl index_health_bsl index_mental_bsl index_political_bsl index_women_bsl
loc advars_pak index_time_bsl index_health_bsl index_political_bsl index_women_bsl
loc advars_gha index_health_bsl index_political_bsl index_women_bsl

use "${dta_working}\\index_hh_vars.dta", clear

foreach var in `hhvars' {
	replace `var'=. if m_`var'==1
}
	
** Panel A: Orthogonality Table by Treatment Status at Baseline, Whole Sample

orth_out `hhvars' using "${tables}\\balance\ttests_pooled.xls", ///
	by(treatment) replace pcompare count se bdec(4) ///
	armlabel(Control Treatment)
	
** Panel B Orthogonality Table by Treatment Status at Baseline, By Country

loc country Ethiopia Ghana Honduras India Pakistan Peru
loc count: word count `country'

forvalues i=1/`count' {
	loc ctry: word `i' of `country'
	orth_out `hhvars' using "${tables}\\balance\ttests_`ctry'.xls" ///
	if country==`i', by(treatment)  pcompare count se bdec(4) ///
	armlabel(Control Treatment) title(`ctry') replace
}

** Regression results, pooled

loc first 1
foreach var in `hhvars' {
	reg `var' treatment i.country
	outreg2 treatment using "${tables}\\balance\balance_reg_pooled.xls", ///
	`=cond(`first'==1, "replace", "")' auto(2) nocons ///
	label stats(coef se pval) cttop(`country')
	loc first 0
}

loc ctries Ethiopia Ghana Honduras India Pakistan Peru
forvalues i=1/6 {
	loc first 1
	loc ctry: word `i' of `ctries'
	foreach var in `hhvars' {
		qui summ `var' if country==`i'
		if r(N) !=0 {
			reg `var' treatment if country==`i'
			outreg2 treatment using "${tables}\\balance\balance_reg_`ctry'.xls", ///
			`=cond(`first'==1, "replace", "")' auto(2) nocons label ///
			stats(coef se pval) cttop(`ctry')
			loc first 0
		}
	}
}

save `temp_hh', replace

use "${dta_working}\\index_mb_vars.dta", replace

foreach var in index_time index_health index_mental index_political index_women {
	replace `var'_bsl=. if m_`var'_bsl==1
	}

save `temp_adult'	

collapse index_time_bsl index_health_bsl index_mental_bsl index_political_bsl index_women_bsl, by(id_hh)
rename id_hh id

merge 1:1 id using `temp_hh', gen(merge_ind)
ta merge_ind // 23 don't perfectly match, 2 only have adult level and 21 only have household level

save `temp_merged'	
	
use `temp_adult'	

** Panel A: Orthogonality Table by Treatment Status at Baseline, Whole Sample
	orth_out `advars' using "${tables}\\balance\ttests_pooled.xls", by(treatment) vappend pcompare count se bdec(4) ///
	armlabel(Control Treatment)


** Panel B Orthogonality Table by Treatment Status at Baseline, By Country
	loc country Ethiopia Ghana Honduras India Pakistan Peru
	loc count: word count `country'
	
	forvalues i=1/`count' {
		loc ctry: word `i' of `country'
		if `i'==5 {
			orth_out `advars_pak' using "${tables}\\balance\ttests_`ctry'.xls" if country==`i', by(treatment) vappend pcompare count se bdec(4) ///
				armlabel(Control Treatment) title(`ctry')
		}
		else if `i' == 2 {
			orth_out `advars_gha' using "${tables}\\balance\ttests_`ctry'.xls" if country==`i', by(treatment) vappend pcompare count se bdec(4) ///
				armlabel(Control Treatment) title(`ctry')
		}
		else {
			orth_out `advars' using "${tables}\\balance\ttests_`ctry'.xls" if country==`i', by(treatment) vappend pcompare count se bdec(4) ///
				armlabel(Control Treatment) title(`ctry')
		}	
	}

** Regressing Each Balance Var on Treatment, with Country Fixed Effects

*A. Pooled

foreach var in `advars' {
	reg `var' treatment i.country
	outreg2 treatment using "${tables}\\balance\balance_reg_pooled.xls", ///
	auto(2) nocons ///
	label stats(coef se pval) cttop(`country')
}

*B. By Country
loc ctries Ethiopia Ghana Honduras India Pakistan Peru

forvalues i=1/6 {
	di "`i'"
	loc ctry: word `i' of `ctries'
	foreach var in `advars' {
		qui summ `var' if country==`i'
		if r(N) !=0 {
			reg `var' treatment if country==`i'
			outreg2 treatment using "${tables}\\balance\balance_reg_`ctry'.xls", ///
			auto(2) nocons label ///
			stats(coef se pval) cttop(`ctry')
		}
	}
}

use `temp_merged', clear

** Regressing All Balance Vars on Treatment, with Country Fixed Effects
loc first 1
reg treatment `hhvars' `advars' i.country
testparm `hhvars' `advars'
loc Ftest `r(F)'
loc pval `r(p)'
outreg2 treatment using "${tables}\\balance\balance_reg_allvars.xls", replace ///
	addstat("F-test", `Ftest', "p-value", `pval')
	
forvalues i = 1/6 {
	loc outcomes
	foreach var in `hhvars' `advars' {
		qui sum `var' if country==`i'
		if `r(N)' > 0 {
			loc outcomes `outcomes' `var'
		}
	}
	reg treatment `outcomes' if country==`i'
	testparm `outcomes'
	loc Ftest `r(F)'
	loc pval `r(p)'
	outreg2 `outcomes' using "${tables}\\balance\balance_reg_allvars.xls", ///
	addstat("F-test", `Ftest', "p-value", `pval')
}
