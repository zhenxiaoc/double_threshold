/**************************************************************
Non-durable estimates, TUP Science Paper Table 4

Table 4 calculates the cost-benefit of the program, based on non-
durable consumption (rather than total consumption, on the premise
that purchasing assets can be used to finance further consumption).
Therefore, the total numbers of non-durable consumption are not
reported in the paper. This do-file constructs those values, and 
estimates the intention-to-treat values of the program on non-
durable consumption.

Stata version: 12.1
***************************************************************/

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

use "${dta_working}\pooled_hh_postanalysis.dta", clear

/* Some households are missing total consumption, or non-durable
consumption, or both. Non-durable consumption is defined in the 
following way:

(1) Has total consumption, missing non-durable: total consumption =
non-durable consumption
(2) Has durable consumption, missing total: non-durable is missing
(3) Has both: non-durable = total - durable
(4) Has neither: missing
*/


foreach var in ctotal_pcmonth cdurable_pcmonth {
	replace `var'_bsl = . if m_`var'_bsl==1
}


foreach t in bsl end fup {
	loc var cnondurable_pcmonth_`t'
	gen `var' = ctotal_pcmonth_`t' if ///
		mi(cdurable_pcmonth_`t') & !mi(ctotal_pcmonth_`t')
	replace `var' = . if mi(ctotal_pcmonth_`t')
	replace `var' = ctotal_pcmonth_`t' - cdurable_pcmonth_`t' if ///
		!mi(ctotal_pcmonth_`t') & !mi(cdurable_pcmonth_`t')
	assert `var'>=0
}


foreach var in ctotal_pcmonth cdurable_pcmonth {
	replace `var'_bsl = . if m_`var'_bsl==1
}

gen m_cnondurable_pcmonth_bsl = mi(cnondurable_pcmonth_bsl)
replace cnondurable_pcmonth_bsl = 0 if m_cnondurable_pcmonth_bsl==1
	

/* Note that for the regressions, the total consumption short survey dummies are
still used for this regression (see SOM Text 1 for a further discussion of their
purpose) */

foreach t in end fup {
	forvalues i = 1/6 {
		qui areg cnondurable_pcmonth_`t' treatment cnondurable_pcmonth_bsl m_cnondurable_pcmonth_bsl css_?? control_* if ///
		country==`i', absorb(geo_cluster) cluster(rand_unit)
		loc treat = _b[treatment]
		di "`var'_`t', country `i' has `treat'"
	}
}

