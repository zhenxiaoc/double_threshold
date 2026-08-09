/****************************************************************
** Table S13 Bounded Attrition: Manski-Lee Method (2009)
** TUP Science Paper
** Purpose: This do-file conducts bounded attrition analysis
	for TUP Cross-Site Science submission. It uses Manski-Lee treatment
	effect bounds to estimate the upper and lower bounds of the 
	effect of attrition on the study outcomes. 
** Stata version: 12.1

This do-file performs two types of Manski-Lee (M-L) Bound attrition.
M-L tests the best and worst case scenarios by removing the "best" and
"worst" individuals from the group with the higher survey rate to 
preserve the same treatment/control share that exists among the full
sample of individuals. For example, if 95% of treatment households
were surveyed, but only 90% of control households, this procedure runs
the same regressions, but by eliminating the 5% of treatment households
with the highest (lowest) values for the outcome variable of interest,
which represent the lower (upper) bounds for the program impacts.

In the case of our evaluation, there are two ways in which we can 
test M-L bounds. The less extreme version removes the highest (lowest)
individuals from the group surveyed at a higher rate to preserve the
treatment/control share from the pooled sample. The more extreme version
removes the highest (lowest) individuals in each country, to preserve
the treatment/control share that existed at baseline in every country.

*****************************************************************/

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

tempfile bounded_attrition temp_full temp_full1 temp_full2 temp_full3 temp_full4 temp_full5 temp_full6

use "${dta_working}\\index_mb_vars.dta", clear

collapse country treatment *index_health* *index_mental* *index_women* *index_political* *index_time*, by(id_hh) // converting adult-level outcomes to household level
rename id_hh id

merge 1:1 id using "${dta_working}\\index_hh_vars.dta", nogen

loc hhvars index_ctotal index_foodsecurity asset_index ind_fin ind_increv
loc advars index_time index_health index_mental index_political index_women
loc outcomevars `hhvars' `advars'

***********************
** POOLED
***********************
gen order_original = _n // used to preserve original order
** 1. Generating variable that takes on value 1 if surveyed and 0 if attrited
foreach t in end fup {
	recode attrition_`t' (0=1) (1=0), gen(surveyed_`t')
}

foreach t in end fup {
	summ surveyed_`t' if treatment==0 // control
	loc cnum_`t'=r(N)
	summ surveyed_`t' if treatment==1 // treatment
	loc tnum_`t'=r(N)
	display `cnum_`t''
	display `tnum_`t''
	
}

loc seed 24 // setting seed for randomization

// getting overall target proportion of treatment and control households
summ treatment // control: 62.98%; treatment: 37.02% (percent of the whole sample)
loc ttarget_full=r(mean)
display "`ttarget_full'" //.37 3891 treatment
loc ctarget_full = 1-r(mean)
display "`ctarget_full'" //.63 6620 control

// for women's empowerment, don't include India
summ treatment if country!=4
loc ttarget_index_women_fup=r(mean)
display "`ttarget_index_women_fup' no India"
loc ctarget_index_women_fup=1-r(mean)
display "`ctarget_index_women_fup' no India"
// for mental health, don't include Pakistan
summ treatment if country!=5
loc ttarget_index_mental_end=r(mean)
display "`ttarget_index_mental_end' no Pakistan"
loc ctarget_index_mental_end=1-r(mean)
display "`ctarget_index_mental_end' no Pakistan"

foreach t in end fup {
	foreach var in `outcomevars' {
		gen highcut_`var'_`t' = .
		gen lowcut_`var'_`t' = .
		if "`var'_`t'"=="index_women_fup" {
			loc ttarget=`ttarget_`var'_`t''
			loc ctarget=`ctarget_`var'_`t''
			display "`var'_`t' `ttarget'"
			display "`var'_`t' `ctarget'"
		}
		else if "`var'_`t'"=="index_mental_end" {
			loc ttarget=`ttarget_`var'_`t''
			loc ctarget=`ctarget_`var'_`t''
			display "`var'_`t' `ttarget'"
			display "`var'_`t' `ctarget'"
		}
		else {
			loc ttarget = `ttarget_full'
			loc ctarget = `ctarget_full'
			display "`var'_`t' `ttarget_full'"
			display "`var'_`t' `ctarget_full'"
		}
		*1. Calculating number of observations needed to cut (group-dependent)
		count if !missing(`var'_`t')
		loc tot_obs = r(N) // total number of nonmissing observations
		display "`tot_obs'"
		count if !missing(`var'_`t') & treatment==0 // control obs
		loc cont_obs = r(N)
		display "`cont_obs'"
		count if !missing(`var'_`t') & treatment==1 // treatment obs
		loc treat_obs = r(N)
		display "`treat_obs'"
		// control
		qui summ treatment if !missing(`var'_`t')
		** control group trimming
		if `r(mean)' < `ttarget' {
			loc cont_keep = (`treat_obs'*`ctarget')/`ttarget' // the number of control obs to keep
			display "`cont_keep'"
			loc cont_cut = round(`cont_obs'-`cont_keep') // difference between smaller sample and full sample
			display "`cont_cut' obs from control in `var'_`t'"
			loc cvars `cvars' `var'_`t'
			cap drop num
			
			*2. Upper Bounds: Trimming Top of Control Group
			replace highcut_`var'_`t' = 1 if treatment==1 & !missing(`var'_`t') // including all treatment hh's
			gen rel_`var'_`t' = 1 if !mi(`var'_`t') & treatment==0
			set seed `seed'
			gen rand_`var'_`t' = runiform()
			gsort +rel_`var'_`t' -`var'_`t' rand_`var'_`t' // sort in descending order
			gen num = _n // descending order count
			display "`cont_cut'"
			replace highcut_`var'_`t' = 0 if num <=`cont_cut' & treatment==0 & !missing(`var'_`t') & rel_`var'_`t'==1
			replace highcut_`var'_`t' = 1 if num > `cont_cut' & treatment==0 & !missing(`var'_`t') & rel_`var'_`t'==1 // including all control households except for the highest
			cap drop num rand_`var'_`t'
			
			*3. Lower Bounds: Trimming Bottom of Control Group
			sort order_original // restoring original order
			replace lowcut_`var'_`t' = 1 if treatment==1 & !missing(`var'_`t') // including all treatment hh's
			set seed `seed'
			gen rand_`var'_`t' = runiform()
			gsort +rel_`var'_`t' +`var'_`t' rand_`var'_`t'
			gen num = _n //ascending order count
			display "`cont_cut'"
			replace lowcut_`var'_`t' = 0 if num <=`cont_cut' & treatment==0 & !missing(`var'_`t') & rel_`var'_`t'==1
			replace lowcut_`var'_`t' = 1 if num >`cont_cut' & treatment==0 & !missing(`var'_`t') & rel_`var'_`t'==1 // including all control households except for the lowest
			
			sort order_original
			cap drop num rand_`var'_`t'
		}
		
		** treatment group trimming
		else if `r(mean)' > `ttarget' {
			loc treat_keep = (`cont_obs'*`ttarget')/`ctarget' // the proportion of treatment observations needed to cut
			display "`treat_keep'"
			loc treat_cut = round(`treat_obs'-`treat_keep') // need to trim from the treatment group
			display "`treat_cut' obs from treat in `var'_`t'"
			loc tvars `tvars' `var'_`t'
			cap drop num
			
			*2. Upper Bounds: Trimming Bottom of Treatment Group
			sort order_original
			replace highcut_`var'_`t' = 1 if treatment==0 & !missing(`var'_`t') // including all control hh's
			set seed `seed'
			gen rel_`var'_`t' = 1 if treatment==1 & !mi(`var'_`t')
			gen rand_`var'_`t' = runiform()
			gsort +rel_`var'_`t' +`var'_`t' rand_`var'_`t' // sort in ascending order
			gen num = _n // ascending order count
			replace highcut_`var'_`t' = 0 if num <=`treat_cut' & treatment==1 & !missing(`var'_`t') & rel_`var'_`t'==1
			replace highcut_`var'_`t' = 1 if num > `treat_cut' & treatment==1 & !missing(`var'_`t') & rel_`var'_`t'==1 // including all treatment households except for the lowest
			sort order_original
			cap drop num rand_`var'_`t'
			
			*3. Lower Bounds: Trimming Top of Treatment Group
			sort order_original // restoring original order
			replace lowcut_`var'_`t' = 1 if treatment==0 & !missing(`var'_`t') // including all control hh's
			set seed `seed'
			gen rand_`var'_`t' = runiform()
			gsort +rel_`var'_`t' -`var'_`t' rand_`var'_`t' // descending order
			gen num = _n // descending order count
			display "`treat_cut'"
			replace lowcut_`var'_`t' = 0 if num <=`treat_cut' & treatment==1 & !missing(`var'_`t') & rel_`var'_`t'==1
			replace lowcut_`var'_`t' = 1 if num >`treat_cut' & treatment==1 & !missing(`var'_`t') & rel_`var'_`t'==1 // including all treatment households except for the highest
			sort order_original
			cap drop num rand_`var'_`t'
		}
	}
}

// getting overall target proportion of treatment and control households by country
forvalues i=1/6 {
	summ treatment if country==`i' // control: 62.98%; treatment: 37.02% (percent of the whole sample)
	loc ttarget`i'=r(mean)
	display "`ttarget`i'' Country `i'"
	loc ctarget`i' = 1-r(mean)
	display "`ctarget`i'' Country `i'"
}

foreach t in end fup {
	foreach var in `outcomevars' {
		gen chcut_`var'_`t'=.
		gen clcut_`var'_`t'=.
		loc mean
		forvalues i=1/6 {
			cap drop num
			qui summ `var'_`t' if country==`i' // making sure that variable exists for a given round
			if r(N)!=0 {
				*1. Calculating number of observations needed to cut (group-dependent)
				count if !missing(`var'_`t') & country==`i' & !missing(treatment)
				loc tot_obs = r(N) // total number of nonmissing observations by country
				display "`tot_obs' TOTAL country `i'"
				count if !missing(`var'_`t') & country==`i' & treatment==0
				loc cont_obs = r(N) // total nonmissing control group
				display "`cont_obs' CONTROL country `i'"
				count if !missing(`var'_`t') & country==`i' & treatment==1
				loc treat_obs = r(N) // total nonmissing treatment group
				display "`treat_obs' TREATMENT country `i'"
				summ treatment if country==`i' & !missing(`var'_`t')
				loc mean`i'=r(mean)
				** TREATMENT GROUP TRIMMING
				display "`mean`i'' for country `i'"
				display "`ttarget`i'' for country `i'"
				if `mean`i'' > `ttarget`i'' {
					loc treat_keep`i' = (`cont_obs'*`ttarget`i'')/`ctarget`i'' // total number of treatment to keep
					assert `treat_keep`i''<`treat_obs'
					loc treat_cut`i' = round(`treat_obs'-`treat_keep`i'') // number to cut from treatment group by country
					display "`treat_cut`i'' TREATMENT obs removed for `var'_`t' in country `i'"
					
					** Lower (upper treatment group)
					replace clcut_`var'_`t' = 1 if treatment==0 & !missing(`var'_`t') & country==`i' // all control
					gen rel`i'_`var'_`t' = 1 if treatment==1 & country==`i' & !missing(`var'_`t')
					set seed `seed'
					gen rand`i'_`var'_`t' = runiform()
					gsort +rel`i'_`var'_`t' -`var'_`t' rand`i'_`var'_`t' // descending order
					gen num = _n
					replace clcut_`var'_`t' = 1 if treatment==1 & num > `treat_cut`i'' & !missing(`var'_`t') & country==`i' & rel`i'_`var'_`t'==1 // keeping all but the highest T
					replace clcut_`var'_`t' = 0 if treatment==1 & num <= `treat_cut`i'' & !missing(`var'_`t') & country==`i' & rel`i'_`var'_`t'==1 // coding highest T as 0
					cap drop num rand`i'_`var'_`t'
					sort order_original
					** Upper (lower treatment group)
					replace chcut_`var'_`t' = 1 if treatment==0 & !missing(`var'_`t') & country==`i' // all control
					set seed `seed'
					gen rand`i'_`var'_`t'= runiform()
					gsort +rel`i'_`var'_`t' +`var'_`t' rand`i'_`var'_`t' // sort in ascending order
					gen num = _n // ascending order count	
					replace chcut_`var'_`t' = 1 if num > `treat_cut`i'' & treatment==1 & !missing(`var'_`t') & country==`i' & rel`i'_`var'_`t'==1 // keeping all but lowest T
					replace chcut_`var'_`t' = 0 if num <=`treat_cut`i'' & treatment==1 & !missing(`var'_`t') & country==`i' & rel`i'_`var'_`t'==1 // coding lowest T as 0
					//append using `temp_full`i''
					sort order_original
					cap drop num rand`i'_`var'_`t'
				}
				** CONTROL GROUP TRIMMING
				else if `mean`i'' < `ttarget`i'' {
					loc cont_keep`i' = (`treat_obs'*`ctarget`i'')/`ttarget`i'' // total number of control to keep
					assert `cont_keep`i''<`cont_obs'
					loc cont_cut`i' = round(`cont_obs'-`cont_keep`i'') // number to cut from control group by country	
					display "`cont_cut`i'' CONTROL obs removed for `var'_`t' in country `i'"
					** Lower (lower control group)
					sort order_original // restoring original order
					replace clcut_`var'_`t' = 1 if treatment==1 & !missing(`var'_`t') & country==`i' // including all treatment hh's
					gen rel`i'_`var'_`t' = 1 if treatment==0 & country==`i' & !missing(`var'_`t')
					set seed `seed'
					gen rand`i'_`var'_`t'=runiform()
					gsort +rel`i'_`var'_`t' +`var'_`t' rand`i'_`var'_`t' // ascending order
					gen num = _n //ascending order count
					replace clcut_`var'_`t' = 1 if num > `cont_cut`i'' & treatment==0 & !missing(`var'_`t') & country==`i' & rel`i'_`var'_`t'==1 // keeping all but lowest C
					replace clcut_`var'_`t' = 0 if num <= `cont_cut`i'' & treatment==0 & !missing(`var'_`t') & country==`i' & rel`i'_`var'_`t'==1 // coding lowest C as 0
					sort order_original
					cap drop num rand`i'_`var'_`t'
					** Upper (upper control group)
					sort order_original // restoring original order
					replace chcut_`var'_`t' = 1 if treatment==1 & !missing(`var'_`t') & country==`i' // including all treatment hh's
					set seed `seed'
					gen rand`i'_`var'_`t' = runiform()
					gsort +rel`i'_`var'_`t' -`var'_`t' rand`i'_`var'_`t' // descending order
					gen num = _n //descending order count
					replace chcut_`var'_`t' = 1 if num > `cont_cut`i'' & treatment==0 & !missing(`var'_`t') & country==`i' & rel`i'_`var'_`t'==1 // keeing all but highest C
					replace chcut_`var'_`t' = 0 if num <= `cont_cut`i'' & treatment==0 & !missing(`var'_`t') & country==`i' & rel`i'_`var'_`t'==1 // coding highest C as 0
					sort order_original
					cap drop num rand`i'_`var'_`t'
				}
			}
		}
	}
}
// all of these should be (no observations)
foreach var in `outcomevars' {
	foreach t in end fup {
		ta country if missing(highcut_`var'_`t') & !missing(`var'_`t')
		ta country if missing(lowcut_`var'_`t') & !missing(`var'_`t')
		ta country if missing(chcut_`var'_`t') & !missing(`var'_`t')
		ta country if missing(clcut_`var'_`t') & !missing(`var'_`t')
	}
}

foreach var in index_ctotal index_foodsecurity asset_index ind_fin ind_increv index_time index_health index_mental index_political index_women {
	foreach t in end fup {
		forvalues i=1/6 {
		display "Country is `i'!!!"
		summ treatment if country==`i'
		display "`var'_`t' COUNTRY SPECIFIC"
		summ treatment if !missing(`var'_`t') & chcut_`var'_`t'==1 & country==`i'
		summ treatment if !missing(`var'_`t') & clcut_`var'_`t'==1 & country==`i'
		}
	}
}

cap drop rel*

** 6. Re-running regressions using trimmed sample, both ways
	loc loop 1
	loc hhvars index_ctotal index_foodsecurity asset_index ind_fin ind_increv
	loc advars index_time index_health index_mental index_political index_women
	loc outcomevars `hhvars' `advars'

	* making empty matrices to store results
	loc matrix_count: word count `outcomevars'
	foreach t in end fup {
		foreach bound in low high cl ch full {
			matrix Pool_`bound'_`t' = J(`matrix_count', 5, .)
		}
	}

	* running regressions
	foreach var in `outcomevars' {
		foreach t in end fup {
			gen fullcut_`var'_`t' = 1 // including full sample as a check
			foreach bound in low high cl ch full {
				loc ss 0
				** Including short survey controls for questions that were asked at short survey round only (only applies to EL1)
				if inlist("`var'", "index_ctotal", "ind_increv", "ind_fin", "index_mental") {
					if "`t'"=="end" {
						loc ss 1
					}
				}
				areg `var'_`t' treatment `var'_bsl m_`var'_bsl m_country_`var'_bsl control_* ///
				`=cond(`ss'==1, "css_g? css_p? css_h?", "")' if `bound'cut_`var'_`t'==1, absorb(geo_cluster) cluster(rand_unit)
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
				loc n_`loop'_`t' = `e(N)'
				matrix Pool_`bound'_`t'[`loop',1] = `p_`loop'_`t''
				matrix Pool_`bound'_`t'[`loop',2] = `coeff_`loop'_`t''
				matrix Pool_`bound'_`t'[`loop',3] = `se_`loop'_`t''
				matrix Pool_`bound'_`t'[`loop',4] = `df_`loop'_`t''
				matrix Pool_`bound'_`t'[`loop',5] = `n_`loop'_`t''
			}
		}
		loc loop = `loop' + 1
	}

** 5. Hochberg-adjustments, both ways
loc precision 3
foreach t in end fup {
	foreach bound in low high cl ch full {
		clear
		set obs 10
		gen initial_order = _n
		gen varname = ""
		loc varcount: word count `outcomevars'
		forvalues i = 1/`varcount' {
			loc var: word `i' of `outcomevars'
			replace varname = "`var'" if initial_order==`i'
		}
		svmat Pool_`bound'_`t'
		rename Pool_`bound'_`t'1 p_value_`t'
		rename Pool_`bound'_`t'2 itt_`t'
		rename Pool_`bound'_`t'3 se_`t'
		rename Pool_`bound'_`t'4 df_`t'
		rename Pool_`bound'_`t'5 n_`t'
		gen q_`t' = .
		sort p_value_`t'
		qui sum p_value_`t'
		drop if mi(p_value_`t')
		qui sum p_value_`t'
		loc count = `r(N)'
		gen order_`t' = _n
		forvalues qval = .999(-.001).000{
		* First Stage
			* Generate value q'*r/M
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
		
		gen output_order = .
		loc outputvars index_ctotal index_foodsecurity asset_index ind_fin index_time ind_increv index_health index_mental index_political index_women
		forvalues i = 1/10 {
			loc output: word `i' of `outputvars'
			replace output_order = `i' if varname=="`output'"
		}
		sort output_order

		outsheet varname itt_output_`t' se_output_`t' p_value_`t' q_`t' n_`t' using "${tables}\\bounded_attrition\Pooled_`bound'_`t'.xls", replace
	}
}
exit
clear
