**************************************
** Set Globals Do-File
** TUP Science Paper
** Stata version: 12.1
**************************************

/* 
Note that this code assumes that you have not changed the underlying folder structure of the data downloaded from
the IPA dataverse. If you have only downloaded some of the do-files, or have changed the folder structure, you will
need to change your globals before continuing.
*/

clear all
set maxvar 15000
version 12.1

cap cd "${path}"

if "${path}"!="" {
	global do "$path\\dofiles\"
	global log "$path\log\"
	global dir "$path\"

	global dta "$path\\data_original"
	global dta_working "$path\\data_modified"

	global figures "$path\\figures\"
	global tables "$path\\tables\"
	global master_test 1
}

if "${path}"=="" | "${path}"=="YOURPATH\ScienceDataRelease" {
	global do "..\dofiles\"
	global log "..\log\"
	global dir "..\"

	global dta "..\data_original"
	global dta_working "..\data_modified"

	global figures "..\figures\"
	global tables "..\tables\"
	global master_test 1
}

