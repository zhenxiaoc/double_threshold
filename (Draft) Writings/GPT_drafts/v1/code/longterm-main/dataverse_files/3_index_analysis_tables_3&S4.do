/* 
Multiple Inference Testing for TUP Pooled Analysis (Table 3, Tables S4)
Purpose: This do file conducts p-value adjustments to account for 
	multiple-inference testing in TUP Pooled Analysis across 10 hypotheses (outlined below):
	total consumption
	food security
	asset index
	financial inclusion
	use of time
	income and revenues
	physical health
	mental health
	political involvement
	women's empowerment
Stata version used: 12.1

* Table of Contents
1. Creating Standardized indices, household
2. Running regressions, household (pooled and country-specific)
3. Member-level regressions and standardized indices (pooled and country-specific)
4. Multiple-inference adjustments, pooled
	storing effect size data for figures
5. Multiple-inferenece adjustments, country-by-country
	storing effect size data for figures
*/

***************************************
*Part I: Creating Standardized Indices
***************************************

*clear all
*set maxvar 15000
*cap log close
*version 12.1

if "${master_test}!"!="1" {
	cap do "1_set_globals.do"
	if _rc==601 {
		di "Please run the 1_set_globals do-file before continuing"
	}
}

tempfile tmp_hh_full tmp2	
use "${dta_working}\pooled_hh.dta", clear	

** Storing outcome variables in local
local countries 1 2 3 4 5 6 // Ethiopia, Ghana, Honduras, India, Pakistan, Peru
loc hhvars index_ctotal ind_increv asset_index index_foodsecurity ind_fin
loc adultvars index_time index_health index_mental index_political index_women
loc outcomevars `hhvars' `adultvars'
	
** Financial Inclusion 
foreach t in bsl end fup {
	foreach var in loan_totalamt sav_totalamt sav_depositamt {
	gen z_`var'_`t' = .
	forvalues i = 1/6 {
		qui sum `var'_`t' if treatment==0 & country==`i' `=cond("`t'"=="bsl", "& m_`var'_`t'==0", "")'
		if `r(N)' > 0 {
			replace z_`var'_`t' = (`var'_`t' - `r(mean)')/`r(sd)' if country==`i' `=cond("`t'"=="bsl", "& m_`var'_`t'==0", "")'
			}
		}
	}
	egen ind_fin_`t' = rowmean(z_loan_*_`t' z_sav_*_`t')
	forvalues i = 1/6 {
		qui sum ind_fin_`t' if treatment==0 & country==`i' `=cond("`t'"=="bsl","&!mi(ind_fin_bsl)", "")'
		if `r(N)' > 0 {
			replace ind_fin_`t' = (ind_fin_`t' - `r(mean)')/`r(sd)' if country==`i'
			}
		}
	}

** Income and Revenue Index
cap drop ind_increv* z_ranimals* z_iagr* z_ibusiness* z_ipaid*
foreach t in bsl end fup {
	foreach var in ranimals_month iagri_month ibusiness_month ipaidlabor_month {
		gen z_`var'_`t' = .
		forvalues i=1/6 {
			summ `var'_`t' if treatment==0 & country==`i' `=cond("`t'"=="bsl", "& m_`var'_`t'==0", "")'
			if `r(N)' > 0 {
				replace z_`var'_`t' = (`var'_`t' - `r(mean)')/`r(sd)' if country==`i' `=cond("`t'"=="bsl", "&m_`var'_`t'==0", "")'
				}
			}
		}
	egen ind_increv_`t' = rowmean(z_ranim*_`t' z_iagri*_`t' z_ibus*_`t' z_ipaid*_`t')
	forvalues i = 1/6 {
		sum ind_increv_`t' if treatment==0 & country==`i' `=cond("`t'"=="bsl", "& !mi(ind_increv_bsl)", "")'
		if `r(N)' > 0 {	
			replace ind_increv_`t' = (ind_increv_`t' - `r(mean)')/`r(sd)' if country==`i'
		}
	}
}

** Total Consumption Index (standardizing total consumption)
cap drop index_ctotal*
foreach t in bsl end fup {
	gen index_ctotal_`t' = .
	forvalues i = 1/6 {
		qui sum ctotal_pcmonth_`t' if treatment==0 & country==`i' `=cond("`t'"=="bsl", "& m_ctotal_pcmonth_`t'==0", "")'
		if `r(N)' > 0 {
			replace index_ctotal_`t' = (ctotal_pcmonth_`t' - `r(mean)')/`r(sd)' if country==`i' `=cond("`t'"=="bsl", "& m_ctotal_pcmonth_`t'==0", "")'
		}
	}
}

***********************************************************************
** Part II. Running regressions and storing results for multiple-inference testing adjustment
***********************************************************************

/* Generating (1) dummy variable for missing outcome 
index for entire country at baseline, and (2) dummy 
variable for missing outcome index for some observations at baseline */
foreach outcomevar in `hhvars' {
	cap confirm variable m_country_`outcomevar'_bsl
	if _rc!=0 {
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
	cap confirm variable m_`outcomevar'_bsl
	if _rc!=0 {
		gen m_`outcomevar'_bsl = (mi(`outcomevar'_bsl) & m_country_`outcomevar'_bsl==0)
		replace `outcomevar'_bsl = 0 if m_`outcomevar'_bsl==1 | m_country_`outcomevar'_bsl==1
	}
}

loc loop 1

** Setting up empty matrices to store results
loc matrix_count: word count `outcomevars'
foreach t in end fup {
	// Pooled level (Table 3)
	matrix Pool_`t' = J(`matrix_count', 6, .)
	// Country level (Tables S4)
	forvalues i = 1/6 {
		matrix C`i'_`t' = J(`matrix_count', 4, .)
	}
}

** Household-variables, pooled regressions
foreach var in `hhvars' {
	foreach t in end fup {
		loc ss 0
		** Including short survey controls for questions that were asked at short survey round only (only applies to EL1)
		if inlist("`var'", "index_ctotal", "ind_increv", "ind_fin") {
			if "`t'"=="end" {
				loc ss 1
			}
		}
		// storing f-test results for equality of treatment across countries
		areg `var'_`t' treat_country? `var'_bsl m_`var'_bsl m_country_`var'_bsl control_* ///
		`=cond(`ss'==1, "css_g? css_p? css_h?", "")', absorb(geo_cluster) cluster(rand_unit)
		test treat_country1==treat_country2==treat_country3==treat_country4==treat_country5==treat_country6
		loc eqtreat_F_`t' `r(F)'
		loc eqtreat_p_`t' `r(p)'
		// storing regression results
		areg `var'_`t' treatment `var'_bsl m_`var'_bsl m_country_`var'_bsl control_* ///
		`=cond(`ss'==1, "css_g? css_p? css_h?", "")', absorb(geo_cluster) cluster(rand_unit)
		//saving values
		matrix b = e(b)
		matrix V = e(V)
		loc coeff_`loop'_`t' = b[1,1]
		loc se_`loop'_`t' = sqrt(V[1,1])
		loc tstat = `coeff_`loop'_`t''/`se_`loop'_`t''
		disp `tstat'
		loc pval = 2*ttail(`e(df_r)', abs(`tstat'))
		loc p_`loop'_`t' = `pval'
		loc df_`loop'_`t' = `e(df_r)'
		matrix Pool_`t'[`loop',1] = `p_`loop'_`t'' // p-value
		matrix Pool_`t'[`loop',2] = `coeff_`loop'_`t'' // itt coefficient
		matrix Pool_`t'[`loop',3] = `se_`loop'_`t'' // itt se
		matrix Pool_`t'[`loop',4] = `df_`loop'_`t'' // degrees of freedom
		matrix Pool_`t'[`loop',5] = `eqtreat_F_`t'' // F-test equality of treatment
		matrix Pool_`t'[`loop',6] = `eqtreat_p_`t'' // p-value equality of treatment
	}
	loc loop = `loop' + 1
}

** Country-specific regressions, household-level variables
forvalues i = 1/6 {
	loc loop`i' 1
	foreach var in `hhvars' {
		foreach t in end fup {
			loc ss 0
			** Including short survey controls for questions that were asked at short survey round only (only applies to EL1)
			if inlist("`var'", "index_ctotal", "ind_increv", "ind_fin") {
				if "`t'"=="end" {
					loc ss 1
				}
			}
			qui sum `var'_`t' if country==`i'
			if `r(N)' > 0 {
				di "`var'_`t' for country `i'"
				areg `var'_`t' treatment `var'_bsl m_`var'_bsl m_country_`var'_bsl control_* ///
				`=cond(`ss'==1, "css_g? css_p? css_h?", "")' if country==`i' ///
				, absorb(geo_cluster) cluster(rand_unit)
				//saving values
				matrix b = e(b)
				matrix V = e(V)
				loc `i'_coeff_`loop`i''_`t' = b[1,1]
				loc `i'_se_`loop`i''_`t' = sqrt(V[1,1])
				loc `i'_tstat = ``i'_coeff_`loop`i''_`t''/``i'_se_`loop`i''_`t''
				disp ``i'_tstat'
				loc `i'_pval = 2*ttail(`e(df_r)', abs(``i'_tstat'))
				loc `i'_p_`loop`i''_`t' = ``i'_pval'
				loc `i'_df_`loop`i''_`t' = `e(df_r)'
				matrix C`i'_`t'[`loop`i'',1] = ``i'_p_`loop`i''_`t'' // p-value
				matrix C`i'_`t'[`loop`i'',2] = ``i'_coeff_`loop`i''_`t'' // itt coefficient
				matrix C`i'_`t'[`loop`i'',3] = ``i'_se_`loop`i''_`t'' // itt se
				matrix C`i'_`t'[`loop`i'',4] = ``i'_df_`loop`i''_`t'' // degrees of freedom
			}
		}
		loc loop`i' = `loop`i'' + 1
	}
}

save  "${dta_working}\index_hh_vars", replace
save `tmp_hh_full'	

**********************************************************************
*** Part III. Member-level indices and regressions, pooled and country
**********************************************************************

use "${dta_working}\pooled_mb", clear

rename *mentalhealth* *mental* // shortening variable names for Stata's length requirement

** Time use - standardizing total time spent on productive activities
cap drop index_time* 
foreach t in bsl end fup {
	gen index_time_`t' = .
	forvalues i = 1/6 {
		sum time_work_`t' if treatment==0 & country==`i' `=cond("`t'"=="bsl", "& m_time_work_bsl==0", "")'
		if `r(N)' > 0 {
			replace index_time_`t' = (time_work_`t' - `r(mean)')/`r(sd)' if country==`i' `=cond("`t'"=="bsl", "& m_time_work_bsl==0", "")'
		}
	}
}

/* Generating (1) dummy variable for missing outcome 
index for entire country at baseline, and (2) dummy 
variable for missing outcome index for some observations at baseline */
foreach outcomevar in `adultvars' {
	cap confirm variable m_country_`outcomevar'_bsl 
	if _rc != 0 {
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
	cap confirm variable m_`outcomevar'_bsl
	if _rc != 0 {
		gen m_`outcomevar'_bsl = 0
		replace m_`outcomevar'_bsl = 1 if mi(`outcomevar'_bsl)
		replace `outcomevar'_bsl = 0 if m_`outcomevar'_bsl==1 | m_country_`outcomevar'_bsl==1
	}
}

** Pooled level regressions, individual level variables
foreach var in `adultvars' {
	foreach t in end fup {
		loc ss 0
		** Including short survey controls for questions that were asked at short survey round only (only applies to EL1)
		if "`var'"=="index_mental" {
			if "`t'"=="end" {
				loc ss 1
			}
		}
		// storing f-test results for equality of treatment across countries 
		areg `var'_`t' treat_country? `var'_bsl m_`var'_bsl m_country_`var'_bsl control_* ///
		`=cond(`ss'==1, "css_g? css_p? css_h?", "")', absorb(geo_cluster) cluster(rand_unit)
		test treat_country1==treat_country2==treat_country3==treat_country4==treat_country5==treat_country6
		loc eqtreat_F_`t' `r(F)'
		loc eqtreat_p_`t' `r(p)'
		// storing regression results
		areg `var'_`t' treatment `var'_bsl m_`var'_bsl m_country_`var'_bsl control_* ///
		`=cond(`ss'==1, "css_g? css_p? css_h?", "")', absorb(geo_cluster) cluster(rand_unit)
		//saving values
		matrix b = e(b)
		matrix V = e(V)
		loc coeff_`loop'_`t' = b[1,1]
		loc se_`loop'_`t' = sqrt(V[1,1])
		loc tstat = `coeff_`loop'_`t''/`se_`loop'_`t''
		disp `tstat'
		loc pval = 2*ttail(`e(df_r)', abs(`tstat'))
		loc p_`loop'_`t' = `pval'
		loc df_`loop'_`t' = `e(df_r)'
		matrix Pool_`t'[`loop',1] = `p_`loop'_`t'' // p-value
		matrix Pool_`t'[`loop',2] = `coeff_`loop'_`t'' // itt coefficient
		matrix Pool_`t'[`loop',3] = `se_`loop'_`t'' // itt se
		matrix Pool_`t'[`loop',4] = `df_`loop'_`t'' // degrees of freedom
		matrix Pool_`t'[`loop',5] = `eqtreat_F_`t'' // F-test equality of treatment
		matrix Pool_`t'[`loop',6] = `eqtreat_p_`t'' // p-value equality of treatment
	}
	loc loop = `loop' + 1
}

** Country level regressions, individiual level variables
forvalues i = 1/6 {
	foreach var in `adultvars' {
		foreach t in end fup {
			loc ss 0
			** Including short survey controls for questions that were asked at short survey round only (only applies to EL1)
			if "`var'"=="index_mental" {
				if "`t'"=="end" {
					loc ss 1
				}
			}
			qui sum `var'_`t' if country==`i'
			if `r(N)' > 0 {
				areg `var'_`t' treatment `var'_bsl m_`var'_bsl m_country_`var'_bsl control_* ///
				`=cond(`ss'==1, "css_g? css_p? css_h?", "")' if country==`i', absorb(geo_cluster) cluster(rand_unit)
				//saving values
				matrix b = e(b)
				matrix V = e(V)
				loc `i'_coeff_`loop`i''_`t' = b[1,1]
				loc `i'_se_`loop`i''_`t' = sqrt(V[1,1])
				loc `i'_tstat = ``i'_coeff_`loop`i''_`t''/``i'_se_`loop`i''_`t''
				disp ``i'_tstat'
				loc `i'_pval = 2*ttail(`e(df_r)', abs(``i'_tstat'))
				loc `i'_p_`loop`i''_`t' = ``i'_pval'
				loc `i'_df_`loop`i''_`t' = `e(df_r)'
				matrix C`i'_`t'[`loop`i'',1] = ``i'_p_`loop`i''_`t'' // p-value
				matrix C`i'_`t'[`loop`i'',2] = ``i'_coeff_`loop`i''_`t'' // itt coefficient
				matrix C`i'_`t'[`loop`i'',3] = ``i'_se_`loop`i''_`t'' // itt se
				matrix C`i'_`t'[`loop`i'',4] = ``i'_df_`loop`i''_`t'' // degrees of freedom
			}
		}
		loc loop`i' = `loop`i'' + 1
	}
}


local countries 1 2 3 4 5 6 // Ethiopia, Ghana, Honduras, India, Pakistan, Peru
loc hhvars index_ctotal ind_increv asset_index index_foodsecurity ind_fin
loc adultvars index_time index_health index_mental index_political index_women
loc outcomevars `hhvars' `adultvars'

save  "${dta_working}\index_mb_vars", replace


***********************************************************
**Part IV: Adjusting P-values for Multiple Hypotheses, Pooled
***********************************************************

/* This code borrows heavily from code used in Michael Anderson's paper "Multiple Inference and Gender Differences
in the Effects of Early Intervention: A Reevaluation of the Abecedarian, Perry Preschool, and Early Training 
Projects." This code can be found at (download): http://are.berkeley.edu/~mlanderson/downloads/fdr_qvalues.do.zip */

loc precision 3 // setting number of significant digits

foreach t in end fup {
	clear
	set obs 10
	gen initial_order = _n
	gen varname = ""
	loc varcount: word count `outcomevars'
	forvalues i = 1/`varcount' {
		loc var: word `i' of `outcomevars'
		replace varname = "`var'" if initial_order==`i'
	}
	svmat Pool_`t'
	rename Pool_`t'1 p_value_`t'
	rename Pool_`t'2 itt_`t'
	rename Pool_`t'3 se_`t'
	rename Pool_`t'4 df_`t'
	rename Pool_`t'5 f_equalitytreat_`t'
	rename Pool_`t'6 p_equalitytreat_`t'
	gen q_`t' = .
	* Sorting by p-value (ascending order)
	sort p_value_`t'
	qui sum p_value_`t'
	drop if mi(p_value_`t')
	qui sum p_value_`t'
	* Collecting total number of p-values tested
	loc count = `r(N)'
	* Collecting rank of unadjusted p-values
	gen order_`t' = _n
	
	/* Set up a loop that begins by checking which hypotheses are rejected at q = .999, 
	then checks which hypotheses are rejected at q = 0.998, then checks which hypotheses are rejected at q = 0.997, etc.  
	The loop ends by checking which hypotheses are rejected at q = 0.000. 
	*/
	forvalues qval = .999(-.001).000{
		* Generate value q'*r/M, where r is the rank and M is the total number of hypotheses being tested
		gen fdr_temp1 = `qval'*order_`t'/`count'
		* Generate binary variable checking condition p(r) <= q'*r/M
		gen reject_temp1 = (fdr_temp1>=p_value_`t') if p_value_`t'~=.
		* Generate variable containing p-value ranks for all p-values that meet above condition
		gen reject_rank1 = reject_temp1*order_`t'
		* Record the rank of the largest p-value that meets above condition
		egen total_rejected1 = max(reject_rank1)
		* A p-value has been rejected at level q if its rank is less than or equal to the rank of the max p-value that meets the above condition
		replace q_`t' = `qval' if order_`t' <= total_rejected1 & order_`t'~=.
		drop fdr_temp* reject_temp* reject_rank* total_rejected*
		sort initial_order
	}
	/* Set up a loop that adjusts p-values for equality of treatment effects across countries.
	Checks which hypotheses are rejected at q = .999, then checks which are rejected at q=.998, etc.
	The loop ends by checking which hypotheses are rejected at q=0.000
	*/
	gen q_equal_`t' = .
	* Sorting ascending order, unadjusted p-values
	sort p_equalitytreat_`t'
	qui sum p_equalitytreat_`t'
	drop if mi(p_equalitytreat_`t')
	qui sum p_equalitytreat_`t'
	* Collecting total number of hypotheses tested
	loc count `r(N)'
	* Collecting rank of unadjusted p-values
	gen order_eq_`t' = _n
	forvalues qval = .999(-.001).000{
	* Generate value q'*r/M, where r is the rank and M is the total number of hypotheses being tested
		gen fdr_temp1 = `qval'*order_eq_`t'/`count'
		* Generate binary variable checking condition p(r) <= q'*r/M
		gen reject_temp1 = (fdr_temp1>=p_equalitytreat_`t') if p_equalitytreat_`t'!=.
		* Generate variable containing p-value ranks for all p-values that meet above condition
		gen reject_rank1 = reject_temp1*order_eq_`t'
		* Record the rank of the largest p-value that meets above condition
		egen total_rejected1 = max(reject_rank1)
		* A p-value has been rejected at level q if its rank is less than or equal to the rank of the max p-value that meets the above condition
		replace q_equal_`t' = `qval' if order_eq_`t' <= total_rejected1 & order_eq_`t'!=.
		drop fdr_temp* reject_temp* reject_rank* total_rejected*
		sort initial_order
	}
	
	* Adding significance stars to the unadjusted p-values for each outcome
	gen stars_`t' = ""
	replace stars_`t' = "***" if p_value_`t' < .01
	replace stars_`t' = "**" if p_value_`t' > .01 & p_value_`t' < .05
	replace stars_`t' = "*" if p_value_`t' > .05 & p_value_`t' < .10
	foreach var in itt se {
		gen temp_`var'_`t' = string(`var'_`t', "%9.`precision'f")
	}
	gen itt_output_`t' = temp_itt_`t' + stars_`t'
	gen se_output_`t' = "(" + temp_se_`t' + ")"
	drop temp_*
	
	* Sorting the data, which now has significance stars and q-values, by the order in published tables
	gen output_order = .
		loc outputvars index_ctotal index_foodsecurity asset_index ind_fin index_time ind_increv index_health index_mental index_political index_women
		forvalues i = 1/10 {
			loc output: word `i' of `outputvars'
			replace output_order = `i' if varname=="`output'"
		}
	sort output_order
	
	* Saving table in Excel
	outsheet varname itt_output_`t' se_output_`t' p_value_`t' q_`t' f_equalitytreat_`t' ///
	q_equal_`t' using "${tables}\\family_indices\pooled_`t'.xls", replace
}


*** Creating Effect Size dataset for family outcomes, pooled dataset

loc hhvars index_ctotal ind_increv asset_index index_foodsecurity ind_fin
loc adultvars index_time index_health index_mental index_political index_women
loc outcomevars `hhvars' `adultvars'
clear

gen varname = ""
foreach t in end fup {
	svmat Pool_`t'
	rename Pool_`t'1 p_`t' // p-value
	rename Pool_`t'2 itt_`t' // itt coefficient
	rename Pool_`t'3 se_`t' // itt se
	rename Pool_`t'4 df_`t' // degrees of freedom
	loc varcount: word count `outcomevars'
	forvalues j = 1/`varcount' {
		loc var: word `j' of `outcomevars'
		replace varname = "`var'" if _n==`j'
	}
}
tempfile effectsize
save `effectsize'
clear

use `effectsize'

loc outcomes index_ctotal index_foodsecurity asset_index ind_fin index_time ind_increv index_health index_mental index_political index_women
loc varcount: word count `outcomes'

gen order = .
forvalues i = 1/`varcount' {
	loc outcome: word `i' of `outcomes'
	replace order = `i' if varname=="`outcome'"
}
sort order
drop order
keep varname *_end *_fup // dropping matrix results Pool_`t'5 and Pool_`t'6 (equality of treatments F-test and corresponding p-value)
	
save "${dta_working}\pooled_family_matrix.dta", replace

*****************************************************************************************
**** Part V. Adjusting P-values for multiple hypothesis adjustment, Country-by-Country
*****************************************************************************************
local countries 1 2 3 4 5 6 // Ethiopia, Ghana, Honduras, India, Pakistan, Peru
loc hhvars index_ctotal ind_increv asset_index index_foodsecurity ind_fin
loc adultvars index_time index_health index_mental index_political index_women
loc outcomevars `hhvars' `adultvars'

* looping through each country
forvalues k = 1/6 {
	* looping through each survey round
	foreach t in end fup {
		clear
		set obs 10
		gen initial_order = _n
		gen varname = ""
		loc varcount: word count `outcomevars'
		forvalues i = 1/`varcount' {
			loc var: word `i' of `outcomevars'
			replace varname = "`var'" if initial_order==`i'
		}
		svmat C`k'_`t'
		rename C`k'_`t'1 p_value_`t'
		rename C`k'_`t'2 itt_`t'
		rename C`k'_`t'3 se_`t'
		rename C`k'_`t'4 df_`t'
		gen q_`t' = .
		* Sorting in ascending order, unadjusted p-values
		sort p_value_`t'
		qui sum p_value_`t'
		drop if mi(p_value_`t')
		qui sum p_value_`t'
		* Collecting total number of hypotheses tested
		loc count = `r(N)'
		* Collecting rank order of unadjusted p-values
		gen order_`t' = _n
		/* Set up a loop that begins by checking which hypotheses are rejected at q = .999, 
		then checks which hypotheses are rejected at q = 0.998, then checks which hypotheses are rejected at q = 0.997, etc.  
		The loop ends by checking which hypotheses are rejected at q = 0.000. 
		*/
		forvalues qval = .999(-.001).000{
			* Generate value q'*r/M, where r is the rank and M is the total number of hypotheses tested
			gen fdr_temp1 = `qval'*order_`t'/`count'
			* Generate binary variable checking condition p(r) <= q'*r/M
			gen reject_temp1 = (fdr_temp1>=p_value_`t') if p_value_`t'~=.
			* Generate variable containing p-value ranks for all p-values that meet above condition
			gen reject_rank1 = reject_temp1*order_`t'
			* Record the rank of the largest p-value that meets above condition
			egen total_rejected1 = max(reject_rank1)
			* A p-value has been rejected at level q if its rank is less than or equal to the rank of the max p-value that meets the above condition
			replace q_`t' = `qval' if order_`t' <= total_rejected1 & order_`t'~=.
			drop fdr_temp* reject_temp* reject_rank* total_rejected*
			sort initial_order
		}
		* Adding significance stars to the unadjusted p-values for each outcome
		gen stars_`t' = ""
		replace stars_`t' = "***" if p_value_`t' < .01
		replace stars_`t' = "**" if p_value_`t' > .01 & p_value_`t' < .05
		replace stars_`t' = "*" if p_value_`t' > .05 & p_value_`t' < .10
		foreach var in itt se {
			gen temp_`var'_`t' = string(`var'_`t', "%9.`precision'f")
		}
		gen itt_output_`t' = temp_itt_`t' + stars_`t'
		gen se_output_`t' = "(" + temp_se_`t' + ")"
		drop temp_*
		
		* Sorting the data, which now has significance stars and q-values, by the order in published tables
		gen output_order = .
		loc outputvars index_ctotal index_foodsecurity asset_index ind_fin index_time ind_increv index_health index_mental index_political index_women
		forvalues i = 1/10 {
			loc output: word `i' of `outputvars'
			replace output_order = `i' if varname=="`output'"
		}
		sort output_order
		
		* Saving table in Excel 
		outsheet varname itt_output_`t' se_output_`t' p_value_`t' q_`t' using "${tables}\\family_indices\Country`k'_`t'.xls", replace
	}
}

*** Creating effect size dataset for family outcomes, country level

local countries 1 2 3 4 5 6
loc hhvars index_ctotal ind_increv asset_index index_foodsecurity ind_fin
loc adultvars index_time index_health index_mental index_political index_women
loc outcomevars `hhvars' `adultvars'

clear
local countries Ethiopia Ghana Honduras India Pakistan Peru

forvalues i = 1/6 {
	loc country: word `i' of `countries'
	gen varname = ""
	gen country = ""
	foreach t in end fup {
		svmat C`i'_`t'
		rename C`i'_`t'1 p_`t' // p-value
		rename C`i'_`t'2 itt_`t' // itt coefficient
		rename C`i'_`t'3 se_`t' // itt se
		rename C`i'_`t'4 df_`t' // degrees of freedom
		loc varcount: word count `outcomevars'
		forvalues j = 1/`varcount' {
			loc var: word `j' of `outcomevars'
			replace varname = "`var'" if _n==`j'
		}
		replace country="`country'"
	}
	tempfile effectsize`i'
	save `effectsize`i''
	clear
}

use `effectsize1'

forvalues i = 2/6 {
	append using `effectsize`i''
}

loc outcomes index_ctotal index_foodsecurity asset_index ind_fin index_time ind_increv index_health index_mental index_political index_women
loc varcount: word count `outcomes'

gen order = .
forvalues i = 1/`varcount' {
	loc outcome: word `i' of `outcomes'
	replace order = `i' if varname=="`outcome'"
}
sort country order
drop order
	
save "${dta_working}\country_family_matrix.dta", replace
