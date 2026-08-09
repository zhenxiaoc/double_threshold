/**************************************************************
Quantile regressions, TUP Science Paper Table 5
This do file runs quantile regressions for the family outcomes
Results are in Table 5 of published Science paper
Stata version: 12.1
***************************************************************/

if "${master_test}!"!="1" {
	cap do "1_set_globals.do"
	if _rc==601 {
		di "Please run the 1_set_globals do-file before continuing"
	}
}

loc hhvars index_ctotal ind_increv asset_index index_foodsecurity ind_fin
loc adultvars index_time index_health index_mental index_political index_women

use "${dta_working}\index_hh_vars.dta", clear
* Country dummy variables
tab country, gen(country_)

/* Note that, as outlined in the notes of Table 5, we use a more parsimonious version of controls for
quantile regressions. 

This reflects the fact that quantile regressions rely on a maximum likelihood 
estimator model, and minimize the least absolute value of error terms. To calculate the coefficients 
of controls, it is necessary that at every quantile tested, for every "block", there exists at least 
one observation with a value of the dependent variable greater than the quantile tested, and one 
observation below with a value of the dependent variable less than the quantile tested (since if there
does not exist one value above and one value below, it is impossible to find an equation that minimizes
the absolute value; the equation will not converge. Given our large number of control variables, in many
instances, at least one does not satisfy the necessary condition that in every block at every quantile
there is at least one observation "above" and "below" the quantile. This leads to the regression "failing
to converge".

We therefore use fewer controls for these regressions. We control for country, the baseline value, and a 
dummy for whether or not the observation was missing at baseline.

For one regression, women's empowerment in endline 1 ("endline"), at the 90th percentile, the regression does
not achieve convergence (thus reflecting the dummified nature of the regressions, and that in one country,
the variance of outcomes is small enough that the whole sample is below the 90th percentile). Therefore, 
we omit the country dummies and just report a regression with baseline level controls. This coefficient
is reported in Table 5, with a point estimate of 0.000, meaning the values are equivalent between treatment
and control at this quantile. */


** Household variable quantile regressions
loc first 1

foreach var in `hhvars' {
	foreach t in end fup {
	
		/* Creating percentiles for Kolmogorov-Smirnov Test
		We test the equality of distributions for all variables. However, given that the p-value associated
		with the Kolmogorov-Smirnov tests are 0.000 for all variables, rather than include a column in the
		table, we simply report in the notes that the value is 0.000 for all variables. This can be confirmed
		by checking this code here. */
		
		pctile pct_`var'_treat_`t' = `var'_`t' if treatment==1, nq(100) genp(`var'_p_`t')
		pctile pct_`var'_con_`t' = `var'_`t' if treatment==0, nq(100)
		la var pct_`var'_treat_`t' "Treatment Households"
		la var pct_`var'_con_`t' "Control Households"
		ksmirnov `var'_`t', by(treatment)
		local pval = round(r(p_cor),.01)
		assert `pval'==0.00

		* Quantile regressions
		foreach j in 0.10 0.25 0.50 0.75 0.90 {
			qui sum `var'_`t'
			noisily di "`var' at `t', q = `j'"
			qui sum `var'_`t' if treatment==0
			loc control_mean `r(mean)'
			qui sum `var'_bsl
			loc baseline_mean `r(mean)'
				
			qreg2 `var'_`t' treatment `var'_bsl m_`var'_bsl country_*, q(`j') cluster(rand_unit)
			if _rc==0 {
				# d ;
				outreg2 treatment using "${tables}\\quantiles\qte_hh.xls", `=cond(`first'==1, "replace", "")'
				addstat("Control mean", `control_mean', "Baseline mean", `baseline_mean', "Quantile", `j',
				"p-value K-S", `pval') adec(3) bdec(3)
				addtext("Block variables used", "No", "Controls used", "No") ;
				# d cr
				loc first 0
			}
			
		}
		
	}
}

use "${dta_working}\index_mb_vars.dta", clear
* Creating country dummy variables
tab country, gen(country_)

loc adultvars index_time index_health index_mental index_political index_women

** Member variable quantile regressions
loc first 1
foreach var in `adultvars' {
	foreach t in end fup {
	
		* Percentiles for Kolmogorov-Smirnov Test
		pctile pct_`var'_treat_`t' = `var'_`t' if treatment==1, nq(100) genp(`var'_p_`t')
		pctile pct_`var'_con_`t' = `var'_`t' if treatment==0, nq(100)
		la var pct_`var'_treat_`t' "Treatment Households"
		la var pct_`var'_con_`t' "Control Households"
		ksmirnov `var'_`t', by(treatment)
		local pval = round(r(p_cor),.01)
		assert `pval'==0.000
		
		* Quantile regressions
		foreach j in 0.10 0.25 0.50 0.75 0.90 {

			* Not using the country dummies for Women's Index at endline 1 at the 90th percentile.
			
				/* Non-convergence errors were also displayed for Women's Index at endline 2 at the 90th percentile 
				and Political Involvement Index at endline 2 at the 10th percentile when including the country dummies*/
				
			if (("`t'"=="end" | "`t'"=="fup")  & "`var'"=="index_women" & "`j'"=="0.90") | "`t'"=="fup" & "`var'"=="index_political" & "`j'"=="0.10" {
				di "No country dummies"
				qui sum `var'_`t'
				noisily di "`var' at `t', q = `j'"
				qui sum `var'_`t' if treatment==0
				loc control_mean `r(mean)'
				qui sum `var'_bsl
				loc baseline_mean `r(mean)'
				qreg2 `var'_`t' treatment `var'_bsl m_`var'_bsl, q(`j') cluster(rand_unit)
				# d ;
				outreg2 treatment using "${tables}//quantiles/qte_mb.xls", `=cond(`first'==1, "replace", "")'
				addstat("Control mean", `control_mean', "Baseline mean", `baseline_mean', "p-value K-S",
				`pval', "Quantile", `j') adec(3) bdec(3)
				addtext("Block variables used", "No", "Controls used", "No") ;
				# d cr
				loc first 0	
			}
			else {
				di "Full set"
				qui sum `var'_`t'
				noisily di "`var' at `t', q = `j'"
				qui sum `var'_`t' if treatment==0
				loc control_mean `r(mean)'
				qui sum `var'_bsl
				loc baseline_mean `r(mean)'
				qreg2 `var'_`t' treatment `var'_bsl m_`var'_bsl country_?, q(`j') cluster(rand_unit)
				# d ;
				outreg2 treatment using "${tables}//quantiles/qte_mb.xls", `=cond(`first'==1, "replace", "")'
				addstat("Control mean", `control_mean', "Baseline mean", `baseline_mean', "p-value K-S",
				`pval', "Quantile", `j') adec(3) bdec(3)
				addtext("Block variables used", "No", "Controls used", "No") ;
				# d cr
				loc first 0
			}	
		}
	}
}
