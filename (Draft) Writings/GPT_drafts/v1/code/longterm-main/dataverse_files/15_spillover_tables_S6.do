/* 
Spillover Analysis - Table S14 
*Purpose: This do file conducts p-value adjustments to account for multiple-inference 
testing in TUP Pooled Analysis across 9 hypotheses (outlined below):
	Indices
	Assets: livestock value, durable goods value, land owned
	Income and Revenues: total monthly income
	Consumption: total consumption at hh level
	Food Security: food security index
	Health index: physical health index, mental health index, perceptions
	Financial Inclusion: loans, savings deposits, savings balance
	Time Use: total time use (in last week, hh level)
	Political Involvement: vote, political party, attended village meeting
	Women's Empowerment: decision-making index
* Stata version: 12.1
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

*********************************
*Part I: Creating indices: Household level
*********************************

tempfile tmp_hh_full tmp2	
use "${dta_working}\pooled_hh_postanalysis", clear	

/* We only have village-level randomization in Ghana, Honduras and Peru. We are therefore 
dropping all other observations, so that for the columns involving treatment households,
we restrict our sample to only those within the countries of interest */

keep if inlist(country, 2, 3, 6) // Ghana, Honduras, Peru

local countries 2 3 6
loc hhvars index_ctotal ind_increv asset_index index_foodsecurity ind_fin
loc adultvars index_time index_health index_mental index_political index_women
loc outcomevars `hhvars' `adultvars'
	
*Financial Inclusion

foreach t in bsl end fup {
	foreach var in loan_totalamt sav_totalamt sav_depositamt {
	gen z_`var'_`t' = .
	foreach i in 2 3 6 {
		qui sum `var'_`t' if treatment==0 & country==`i' `=cond("`t'"=="bsl", "& m_`var'_`t'==0", "")'
		if `r(N)' > 0 {
			replace z_`var'_`t' = (`var'_`t' - `r(mean)')/`r(sd)' if country==`i' `=cond("`t'"=="bsl", "& m_`var'_`t'==0", "")'
			}
		}
	}
	egen ind_fin_`t' = rowmean(z_loan_*_`t' z_sav_*_`t')
	foreach i in 2 3 6 {
		qui sum ind_fin_`t' if treatment==0 & country==`i' `=cond("`t'"=="bsl","&!mi(ind_fin_bsl)", "")'
		if `r(N)' > 0 {
			replace ind_fin_`t' = (ind_fin_`t' - `r(mean)')/`r(sd)' if country==`i'
			}
		}
	}

** Index Income and Revenue
cap drop ind_increv* z_ranimals* z_iagr* z_ibusiness* z_ipaid*
foreach t in bsl end fup {
	foreach var in ranimals_month iagri_month ibusiness_month ipaidlabor_month {
		gen z_`var'_`t' = .
		foreach i in 2 3 6 {
			qui summ `var'_`t' if treatment==0 & country==`i' `=cond("`t'"=="bsl", "& m_`var'_`t'==0", "")'
			if `r(N)' > 0 {
				replace z_`var'_`t' = (`var'_`t' - `r(mean)')/`r(sd)' if country==`i' `=cond("`t'"=="bsl", "& m_`var'_`t'==0", "")'
				}
			}
		}
	egen ind_increv_`t' = rowmean(z_ranim*_`t' z_iagri*_`t' z_ibus*_`t' z_ipaid*_`t')
	foreach i in 2 3 6 {
		qui sum ind_increv_`t' if treatment==0 & country==`i' `=cond("`t'"=="bsl", "& !mi(ind_increv_bsl)", "")'
		if `r(N)' > 0 {	
			replace ind_increv_`t' = (ind_increv_`t' - `r(mean)')/`r(sd)' if country==`i'
		}
	}
}

cap drop index_ctotal*
foreach t in bsl end fup {
	gen index_ctotal_`t' = .
	foreach i in 2 3 6 {
		qui sum ctotal_pcmonth_`t' if treatment==0 & country==`i' `=cond("`t'"=="bsl", "& m_ctotal_pcmonth_`t'==0", "")'
		if `r(N)' > 0 {
			replace index_ctotal_`t' = (ctotal_pcmonth_`t' - `r(mean)')/`r(sd)' if country==`i' `=cond("`t'"=="bsl", "& m_ctotal_pcmonth_`t'==0", "")'
		}
	}
}

// Generating missing value dummies for household-level indices
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

*** Getting and storing regresion results: HH Pooled ***
loc matrix_count: word count `outcomevars'

// Making Matrices to include all results (household and adult-level)
foreach k in end fup {
loc loop 1
	foreach treat in treatment spillover_treat villagelevel_treat invillage_treat  {
		matrix Pool_`treat'_`k' = J(`matrix_count', 4, .) // pooled matrices
		foreach i in 2 3 6 {
			matrix C`i'_`treat'_`k' = J(`matrix_count', 4, .) // country-specific matrices
		}
	}
	// Household-level regressions
	foreach var in `hhvars' {
		loc ss 0
		** Including short survey controls for questions that were asked at short survey round only (only applies to EL1)
			if inlist("`var'", "index_ctotal", "ind_increv", "ind_fin") {
				if "`k'"=="end" {
					loc ss 1
				}
			}
		// 1. Treatment 
		* This is the equivalent of Table 3, but restricted to the three countries of interest.
		areg `var'_`k' treatment `var'_bsl m_`var'_bsl m_country_`var'_bsl control_* ///
		`=cond(`ss'==1, "css_g? css_p? css_h?", "")', absorb(geo_cluster) cluster(rand_unit)
		//saving values
		matrix b = e(b)
		matrix V = e(V)
		loc coeff_`loop'_treatment = b[1,1]
		loc se_`loop'_treatment = sqrt(V[1,1])
		loc tstat = `coeff_`loop'_treatment'/`se_`loop'_treatment'
		disp `tstat'
		loc pval = 2*ttail(`e(df_r)', abs(`tstat'))
		loc p_`loop'_treatment = `pval'
		loc df_`loop'_treatment = `e(df_r)'
		matrix Pool_treatment_`k'[`loop',1] = `p_`loop'_treatment'
		matrix Pool_treatment_`k'[`loop',2] = `coeff_`loop'_treatment'
		matrix Pool_treatment_`k'[`loop',3] = `se_`loop'_treatment'
		matrix Pool_treatment_`k'[`loop',4] = `df_`loop'_treatment'
		
		// 2. In-village treatment
		/* This ignores the pure control villages, looking only at households in villages 
		where the program took place */
		areg `var'_`k' invillage_treat `var'_bsl m_`var'_bsl m_country_`var'_bsl control_* ///
		`=cond(`ss'==1, "css_g? css_p? css_h?", "")', absorb(geo_cluster)
		//saving values
		matrix b = e(b)
		matrix V = e(V)
		loc coeff_`loop'_invillage_treat = b[1,1]
		loc se_`loop'_invillage_treat = sqrt(V[1,1])
		loc tstat = `coeff_`loop'_invillage_treat'/`se_`loop'_invillage_treat'
		disp `tstat'
		loc pval = 2*ttail(`e(df_r)', abs(`tstat'))
		loc p_`loop'_invillage_treat = `pval'
		loc df_`loop'_invillage_treat = `e(df_r)'
		matrix Pool_invillage_treat_`k'[`loop',1] = `p_`loop'_invillage_treat'
		matrix Pool_invillage_treat_`k'[`loop',2] = `coeff_`loop'_invillage_treat'
		matrix Pool_invillage_treat_`k'[`loop',3] = `se_`loop'_invillage_treat'
		matrix Pool_invillage_treat_`k'[`loop',4] = `df_`loop'_invillage_treat'
		
		// 3. Village-level treatment
		/* This restricts the control group to only those in pure control villages (ie
		where there is no possibility of spillovers */
		reg `var'_`k' villagelevel_treat `var'_bsl m_`var'_bsl m_country_`var'_bsl control_* ///
		`=cond(`ss'==1, "css_g? css_p? css_h?", "")', cluster(geo)
		//saving values
		matrix b = e(b)
		matrix V = e(V)
		loc coeff_`loop'_villagelevel_treat = b[1,1]
		loc se_`loop'_villagelevel_treat = sqrt(V[1,1])
		loc tstat = `coeff_`loop'_villagelevel_treat'/`se_`loop'_villagelevel_treat'
		disp `tstat'
		loc pval = 2*ttail(`e(df_r)', abs(`tstat'))
		loc p_`loop'_villagelevel_treat = `pval'
		loc df_`loop'_villagelevel_treat = `e(df_r)'
		matrix Pool_villagelevel_treat_`k'[`loop',1] = `p_`loop'_villagelevel_treat'
		matrix Pool_villagelevel_treat_`k'[`loop',2] = `coeff_`loop'_villagelevel_treat'
		matrix Pool_villagelevel_treat_`k'[`loop',3] = `se_`loop'_villagelevel_treat'
		matrix Pool_villagelevel_treat_`k'[`loop',4] = `df_`loop'_villagelevel_treat'
		
		// 4. Spillover
		/* This compares control households in treatment communities to pure control
		villages */
		reg `var'_`k' spillover_treat `var'_bsl m_`var'_bsl m_country_`var'_bsl control_* ///
		`=cond(`ss'==1, "css_g? css_p? css_h?", "")', cluster(geo)
		//saving values
		matrix b = e(b)
		matrix V = e(V)
		loc coeff_`loop'_spillover_treat = b[1,1]
		loc se_`loop'_spillover_treat = sqrt(V[1,1])
		loc tstat = `coeff_`loop'_spillover_treat'/`se_`loop'_spillover_treat'
		disp `tstat'
		loc pval = 2*ttail(`e(df_r)', abs(`tstat'))
		loc p_`loop'_spillover_treat = `pval'
		loc df_`loop'_spillover_treat = `e(df_r)'
		matrix Pool_spillover_treat_`k'[`loop',1] = `p_`loop'_spillover_treat'
		matrix Pool_spillover_treat_`k'[`loop',2] = `coeff_`loop'_spillover_treat'
		matrix Pool_spillover_treat_`k'[`loop',3] = `se_`loop'_spillover_treat'
		matrix Pool_spillover_treat_`k'[`loop',4] = `df_`loop'_spillover_treat'
			
		loc loop = `loop' + 1
	}
}

save  "${dta_working}\index_hh_vars_spillover", replace
save `tmp_hh_full'	


***********************************
**Part II: Adult-level indices
***********************************

use "${dta_working}\pooled_mb_postanalysis", clear

keep if inlist(country, 2, 3, 6) // only Ghana, Honduras and Peru

rename *mentalhealth* *mental*

cap drop index_time* 
foreach t in bsl end fup {
	gen index_time_`t' = .
	foreach i in 2 3 6 {
		sum time_work_`t' if treatment==0 & country==`i' `=cond("`t'"=="bsl", "& m_time_work_bsl==0", "")'
		if `r(N)' > 0 {
			replace index_time_`t' = (time_work_`t' - `r(mean)')/`r(sd)' if country==`i' `=cond("`t'"=="bsl", "& m_time_work_bsl==0", "")'
		}
	}
}

// Generating missing value dummies for adult-level indices
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

encode id_hh, gen(id_num)

//Running Regressions (Adult-level, Pooled)
foreach k in end fup {
loc loop 6
	foreach var in `adultvars' {
		loc ss 0
		** Including short survey controls for questions that were asked at short survey round only (only applies to EL1)
		if "`var'"=="index_mental" {
			if "`k'"=="end" {
				loc ss 1
			}
		}
		//1. Treatment
		* This is the equivalent of Table 3, but restricted to the three countries of interest.
		areg `var'_`k' treatment `var'_bsl m_`var'_bsl m_country_`var'_bsl control_* ///
		`=cond(`ss'==1, "css_g? css_p? css_h?", "")', absorb(geo_cluster) cluster(rand_unit)
		//saving values
		matrix b = e(b)
		matrix V = e(V)
		loc coeff_`loop'_treatment = b[1,1]
		loc se_`loop'_treatment = sqrt(V[1,1])
		loc tstat = `coeff_`loop'_treatment'/`se_`loop'_treatment'
		disp `tstat'
		loc pval = 2*ttail(`e(df_r)', abs(`tstat'))
		loc p_`loop'_treatment = `pval'
		loc df_`loop'_treatment = `e(df_r)'
		matrix Pool_treatment_`k'[`loop',1] = `p_`loop'_treatment'
		matrix Pool_treatment_`k'[`loop',2] = `coeff_`loop'_treatment'
		matrix Pool_treatment_`k'[`loop',3] = `se_`loop'_treatment'
		matrix Pool_treatment_`k'[`loop',4] = `df_`loop'_treatment'
		
		//2. In-village Treatment
		/* This ignores the pure control villages, looking only at households in villages 
		where the program took place */
		areg `var'_`k' invillage_treat `var'_bsl m_`var'_bsl m_country_`var'_bsl control_* ///
		`=cond(`ss'==1, "css_g? css_p? css_h?", "")', absorb(geo_cluster) cluster(id_hh)
		//saving values
		matrix b = e(b)
		matrix V = e(V)
		loc coeff_`loop'_invillage_treat = b[1,1]
		loc se_`loop'_invillage_treat = sqrt(V[1,1])
		loc tstat = `coeff_`loop'_invillage_treat'/`se_`loop'_invillage_treat'
		disp `tstat'
		loc pval = 2*ttail(`e(df_r)', abs(`tstat'))
		loc p_`loop'_invillage_treat = `pval'
		loc df_`loop'_invillage_treat = `e(df_r)'
		matrix Pool_invillage_treat_`k'[`loop',1] = `p_`loop'_invillage_treat'
		matrix Pool_invillage_treat_`k'[`loop',2] = `coeff_`loop'_invillage_treat'
		matrix Pool_invillage_treat_`k'[`loop',3] = `se_`loop'_invillage_treat'
		matrix Pool_invillage_treat_`k'[`loop',4] = `df_`loop'_invillage_treat'
		
		//3. Village-level treatment
		/* This restricts the control group to only those in pure control villages (ie
		where there is no possibility of spillovers */
		reg `var'_`k' villagelevel_treat `var'_bsl m_`var'_bsl m_country_`var'_bsl control_* ///
		`=cond(`ss'==1, "css_g? css_p? css_h?", "")', cluster(geo)
		//saving values
		matrix b = e(b)
		matrix V = e(V)
		loc coeff_`loop'_villagelevel_treat = b[1,1]
		loc se_`loop'_villagelevel_treat = sqrt(V[1,1])
		loc tstat = `coeff_`loop'_villagelevel_treat'/`se_`loop'_villagelevel_treat'
		disp `tstat'
		loc pval = 2*ttail(`e(df_r)', abs(`tstat'))
		loc p_`loop'_villagelevel_treat = `pval'
		loc df_`loop'_villagelevel_treat = `e(df_r)'
		matrix Pool_villagelevel_treat_`k'[`loop',1] = `p_`loop'_villagelevel_treat'
		matrix Pool_villagelevel_treat_`k'[`loop',2] = `coeff_`loop'_villagelevel_treat'
		matrix Pool_villagelevel_treat_`k'[`loop',3] = `se_`loop'_villagelevel_treat'
		matrix Pool_villagelevel_treat_`k'[`loop',4] = `df_`loop'_villagelevel_treat'
		
		//4. Spillovers
		/* This compares control households in treatment communities to pure control
		villages */
		reg `var'_`k' spillover_treat `var'_bsl m_`var'_bsl m_country_`var'_bsl control_* ///
		`=cond(`ss'==1, "css_g? css_p? css_h?", "")', cluster(geo)
		//saving values
		matrix b = e(b)
		matrix V = e(V)
		loc coeff_`loop'_spillover_treat = b[1,1]
		loc se_`loop'_spillover_treat = sqrt(V[1,1])
		loc tstat = `coeff_`loop'_spillover_treat'/`se_`loop'_spillover_treat'
		disp `tstat'
		loc pval = 2*ttail(`e(df_r)', abs(`tstat'))
		loc p_`loop'_spillover_treat = `pval'
		loc df_`loop'_spillover_treat = `e(df_r)'
		matrix Pool_spillover_treat_`k'[`loop',1] = `p_`loop'_spillover_treat'
		matrix Pool_spillover_treat_`k'[`loop',2] = `coeff_`loop'_spillover_treat'
		matrix Pool_spillover_treat_`k'[`loop',3] = `se_`loop'_spillover_treat'
		matrix Pool_spillover_treat_`k'[`loop',4] = `df_`loop'_spillover_treat'
			
		loc loop = `loop' + 1
	}
}

save  "${dta_working}\index_mb_vars_spillover", replace

**** PRECISION *****
loc precision 3
// Pooled P-value adjustment
foreach k in end fup {
	foreach treat in treatment villagelevel_treat invillage_treat spillover_treat {
		clear
		set obs 10
		gen initial_order = _n
		gen varname = ""
		loc varcount: word count `outcomevars'
		forvalues i = 1/`varcount' {
			loc var: word `i' of `outcomevars'
			replace varname = "`var'" if initial_order==`i'
		}
		svmat Pool_`treat'_`k'
		rename Pool_`treat'_`k'1 p_value
		rename Pool_`treat'_`k'2 itt
		rename Pool_`treat'_`k'3 se
		rename Pool_`treat'_`k'4 df
		gen q = .
		sort p_value
		qui sum p_value
		drop if mi(p_value)
		qui sum p_value
		loc count = `r(N)'
		gen order = _n
		forvalues qval = .999(-.001).000{
		* First Stage
		* Generate value q'*r/M
			gen fdr_temp1 = `qval'*order/`count'
			* Generate binary variable checking condition p(r) <= q'*r/M
			gen reject_temp1 = (fdr_temp1>=p_value) if p_value~=.
			* Generate variable containing p-value ranks for all p-values that meet above condition
			gen reject_rank1 = reject_temp1*order
			* Record the rank of the largest p-value that meets above condition
			egen total_rejected1 = max(reject_rank1)
			* A p-value has been rejected at level q if its rank is less than or equal to the rank of the max p-value that meets the above condition
			replace q = `qval' if order <= total_rejected1 & order~=.
			drop fdr_temp* reject_temp* reject_rank* total_rejected*
			sort initial_order
		}
		gen stars = ""
		replace stars = "***" if p_value < .01
		replace stars = "**" if p_value > .01 & p_value < .05
		replace stars = "*" if p_value > .05 & p_value < .10
		foreach var in itt se {
			gen temp_`var' = string(`var', "%9.`precision'f")
		}
		gen itt_output = temp_itt + stars
		gen se_output = "(" + temp_se + ")"
		drop temp_*
		
		gen output_order = .
			loc outputvars index_ctotal index_foodsecurity asset_index ind_fin index_time ind_increv index_health index_mental index_political index_women
			forvalues i = 1/10 {
				loc output: word `i' of `outputvars'
				replace output_order = `i' if varname=="`output'"
			}
		sort output_order
		
		if "`k'"=="end" {
			outsheet varname itt_output se_output p_value q using "${tables}\\spillover\\`treat'.xls", replace
		}
		else if "`k'" == "fup" {
			outsheet varname itt_output se_output p_value q using "${tables}\\spillover\\`treat'_fup.xls", replace
		}
		
	}
}

save "${dta_working}\country_family_matrix_spillover.dta", replace
