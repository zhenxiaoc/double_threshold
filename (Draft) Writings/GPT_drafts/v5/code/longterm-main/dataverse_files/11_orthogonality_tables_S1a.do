*****************************************************************
** Summary Statistics at Baseline by Country
** TUP Cross Site Paper
** Tables S1a
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
	
use "${dta}\\pooled_hh_baseline_sumstats.dta", clear
ta country if mi(treatment) // Peru, Hnd, India hh's not in sample. Ghaan SOUP hh's (not in analysis)
drop if mi(treatment)

** 1. Median per capita monthly consumption
gen medcons = .
forvalues i=1/6 {
	qui summ ctotal_pcmonth_bsl if country==`i', det
	display "Country `i' has `r(p50)' median consumption at baseline"
	replace medcons = `r(p50)' if country==`i'
}
** 2. Percentage of countries below the poverty line of 2005 $1.25/day (2014 dollars is $1.52)
gen dpovertyline = 0
replace dpovertyline = 1 if ctotal_pcday_bsl<=1.52
gen povertyline=.
forvalues i=1/6 {
	summ dpovertyline if country==`i'
	replace povertyline = `r(mean)' if country==`i'
}
bys country: ta dpovertyline
ta povertyline

** 3. Average household size at baseline
gen avghhsize = .
forvalues i=1/6 {
	summ hhsize_bsl if country==`i'
	display "Country `i' has an average household size of `r(mean)'"
	replace avghhsize = `r(mean)' if country==`i'
}

** 4. Average household size - national averages
gen natl_hhsize = .
loc ntlhhsize 4.6 3.7 4.4 4.8 6.8 3.9
forvalues i=1/6 {
	loc nh: word `i' of `ntlhhsize'
	replace natl_hhsize = `nh' if country==`i'
}
ta natl_hhsize
summ natl_hhsize

** 5. Total annual household consumption
gen ctotal_year_bsl = ctotal_pcmonth_bsl*12*hhsize_bsl // total annual household consumption by country
tabstat ctotal_year_bsl, by(country)

** 6. Summary stats for baseline consumption using Deaton's measure (without health or durable spending)
foreach var in chealth_pcmonth_bsl cdurable_pcmonth_bsl {
	replace `var' = -1*`var'
}
** 7. Revised consumption, monthly(medians)
egen ctotalwb_pcmonth_bsl = rowtotal(ctotal_pcmonth_bsl chealth_pcmonth_bsl cdurable_pcmonth_bsl), mis
gen medcons2 = .
forvalues i=1/6 {
	qui summ ctotalwb_pcmonth_bsl if country==`i', det
	replace medcons2 = `r(p50)' if country==`i'
}

tabstat ctotalwb_pcmonth_bsl, by(country) s(med)
gen ctotalwb_pcday_bsl = ctotalwb_pcmonth_bsl/30 // daily
tabstat ctotalwb_pcday_bsl, by(country) s(med)

foreach var in chealth_pcmonth_bsl cdurable_pcmonth_bsl {
	replace `var' = -1*`var'
}

** 8. Revised poverty lines - Percentage of countries below the poverty line of 2005 $1.25/day (2014 dollars is $1.52)
gen dpovertyline2 = 0
replace dpovertyline2 = 1 if ctotalwb_pcday_bsl<=1.52
gen povertyline2=.
forvalues i=1/6 {
	summ dpovertyline2 if country==`i'
	replace povertyline2 = `r(mean)' if country==`i'
}
bys country: ta dpovertyline2
ta povertyline2

** 9. National level poverty lines data
loc natl_plines .368 .286 .165 .236 .127 .029
gen povertyline_natl = .
forvalues i=1/6 {
	loc pl: word `i' of `natl_plines'
	replace povertyline_natl = `pl' if country==`i'
}
ta povertyline_natl
summ povertyline_natl

** 10. Exporting summary stats table for these variables by country
orth_out ctotal_pcmonth_bsl medcons povertyline ctotalwb_pcmonth_bsl medcons2 povertyline2 povertyline_natl ///
using "${tables}\\balance\summarystats_bsl.xlsx", by(country) replace se overall ///
varlabel("Total consumption" "Median consumption" "Povertyline 1" "Modified consumption" "Median modified consumption" "poverty line 2") ///
armlabel("Ethiopia" "Ghana" "Honduras" "India" "Pakistan" "Peru") title("Baseline Summary Statistics, by Country") bd(2)

clear
exit
