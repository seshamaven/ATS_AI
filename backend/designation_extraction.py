# designation_only_extractor.py
"""
Lightweight Designation Extractor
- Purpose: extract the most-likely current/most-recent designation from resume text.
- Approach:
  1. Normalize text.
  2. Cut off at any "Portfolio / Project" section to avoid project titles being mistaken for job titles.
  3. Scan around "Work / Experience" sections when available; fallback to scanning the whole text.
  4. Use a compact known-designations list + fuzzy matching as a primary method.
  5. Use regex heuristics (common patterns like "Senior Consultant") as fallback.
  6. Strictly reject lines that contain known non-title keywords (project, client, application, including, etc.)
- Returns: best-match designation string (title-cased) or None if not found.
"""

import re
from difflib import SequenceMatcher
from typing import Optional, List, Tuple


# -------------------------------------------------------------------------
# CONFIG: adjust or extend as needed
# -------------------------------------------------------------------------
# Small but high-coverage set of common role keywords (kept compact for speed)
COMMON_TITLES = {
    "senior consultant", "consultant", "senior software engineer", "software engineer",
    "software developer", "full stack developer", "fullstack developer", "developer",
    "senior developer", "power platform developer", "power apps developer",
    "power platform consultant", "power apps consultant", "power platform engineer",
    "senior consultant", "technical lead", "technical lead", "team lead",
    "senior consultant", "senior", "consultant", "senior consultant",
    "manager", "project manager", "program manager", "assistant manager",
    "lead", "technical architect", "solutions architect", "business analyst",
    "system administrator", "administrator", "senior consultant", "senior manager",
    "senior engineer", "senior software engineer", "senior consultant", "senior",
    "associate consultant", "associate", "intern", "trainee",
    # Director roles
        "director", "technical director", "director - technology", "director technology", "Director - Technology",
        "director of technology", "it director", "software director", "engineering director",
        "development director", "program director", "project director", "delivery director",
        "senior director", "sr. director", "sr director", "associate director",
        "assistant director", "deputy director", "executive director", "managing director",
        
        # Consultant roles
        "consultant", "senior consultant", "sr. consultant", "sr consultant",
        "executive consultant", "technical consultant", "it consultant", "business consultant",
        "management consultant", "senior technical consultant", "lead consultant",
        "principal consultant", "associate consultant", "junior consultant",
        
        # Manager roles
        "business development executive", "executive consultant", "program manager/director",
        "Program Manager", "ai program manager", "ev program manager", "it program manager",
        "logistics program manager", "material program manager", "materials program manager",
        "program manager", "program manager,", "project/program manager", "sap program manager",
        "sap service management program manager", "sr cybersecurity program manager",
        "sr program manager", "sr. it program manager", "sr. program manager",
        "sr. technical program manager", "strategic program manager", "technical program manager",
        "sr project manager", "sr. project manager",
        "Project Manager", "agile it project manager", "amr/ami project manager",
        "assistant project manager", "associate project manager", "business project manager",
        "communications project manager", "construction project manager",
        "corporate communications project manager", "digital marketing & media project manager",
        "digital marketing project manager", "ems upgrade project manager",
        "healthcare project manager", "information security project manager",
        "infrastructure project manager", "it infrastructure project manager", "it project manager",
        "it project manager,", "jr. project manager", "marketing project manager",
        "marketing project manager/producer", "organizational development project manager",
        "part time project manager", "peoplesoft project manager", "planning project manager",
        "privacy project manager", "product project manager", "program & project manager",
        "project manager", "project manager/administrator", "project manager/program",
        "project manager/systems", "qa project manager", "retail project manager",
        "secops project manager", "sharepoint 2010 project manager",
        "sharepoint technical project manager", "software release project manager",
        "sr cloud finops project manager", "sr it project manager", "sr. hr project manager",
        "sr. infrastructure project manager", "sr. it project manager", "sr.project manager",
        "sw driver development project manager", "technical project manager",
        "Lead", "Technical Lead", "lead", "technical lead",
        "Analyst", "Business Analyst", "adobe cq systems analyst", "agile business analyst",
        "analyst", "application analyst", "application qa analyst", "application support analyst",
        "application system analyst", "application systems analyst", "application systems analyst,",
        "application systems analyst/programmer", "applications systems analyst", "arcos analyst",
        "arcos business analyst/sys", "asset & configuration analyst", "automation qa analyst",
        "bi business analyst", "bi business analyst/", "bi operations analyst", "bi qa analyst",
        "biztalk and sql database analyst", "budget & fiscal analyst", "busines data analyst",
        "business & process improvement analyst", "business analyst", "business analyst/",
        "business analyst/bde", "business analyst/project", "business applications analyst",
        "business data analyst", "business objects analyst", "business operations analyst",
        "business process analyst", "business system analyst", "business systems analyst",
        "business systems analyst/project", "business systems data analyst", "business/qa analyst",
        "c#.netconfiguration analyst", "chinese localization qa analyst",
        "coldfusion application analyst", "coldfusion applications systems analyst",
        "commercial analyst", "compensation analyst", "compliance reporting analyst",
        "computer systems analyst", "consumer market research analyst", "cyber security analyst",
        "cyberark support analyst", "data analyst", "data base programmer/analyst",
        "data governance analyst", "data security analyst", "data visualization analyst",
        "data/ information analyst", "data/information analyst", "database analyst",
        "digital marketing & project analyst", "distributed energy resource analyst",
        "edi business analyst", "energy billing and settlement analyst", "enterprise data analyst",
        "esri arcgis desktop analyst", "etl technical analyst", "financial analyst",
        "financial reporting analyst", "financial risk analyst", "help desk analyst",
        "helpdesk analyst", "hr analyst", "hr bi analyst", "hr data analyst", "hris analyst",
        "hris business systems analyst", "hybris business analyst", "iam analyst",
        "informatics business analyst", "information analyst", "information risk analyst",
        "information security analyst", "information security forensic analyst",
        "information security risk analyst", "information technology analyst",
        "inventory data analyst", "it audit & compliance analyst", "it business analyst",
        "it business systems analyst", "it compliance analyst", "it marketing bi data analyst",
        "it project analyst", "it quality analyst", "it security analyst", "it service desk analyst",
        "it systems analyst", "itam analyst", "java systems analyst", "jr ca siteminder analyst",
        "kronos analyst", "labor relations analyst", "legislative and policy analyst",
        "mainframe application systems analyst", "mainframe production support analyst",
        "market research & channel analyst", "market research analyst",
        "marketing & project analyst", "marketing analyst", "marketplace data analyst",
        "maximo business analyst", "medical device integration analyst",
        "merchandising data analyst", "microsoft sam analyst", "modernization analyst",
        "ms project analyst", "net and sql bi analyst", "net data analyst",
        "network intrusion detection system analyst", "network security analyst",
        "operational data analyst", "operations analyst", "operations data analyst",
        "operations support analyst", "oracle crm analyst", "part time business systems analyst",
        "payroll analyst", "pci compliance/security analyst",
        "peoplesoft benefits administration analyst/subject", "peoplesoft functional analyst",
        "peoplesoft hcm analyst", "peoplesoft hcm business analyst", "performance data analyst",
        "pipeline compliance analyst", "pipeline compliance records analyst",
        "pipeline measurement analyst", "power & performance analyst",
        "privacy and data protection analyst", "product insights analyst", "program analyst",
        "programming analyst", "project controls analyst", "qa analyst", "qa test analyst",
        "qlikview bi analyst", "quality analyst", "quality assurance analyst",
        "quality test analyst", "reporting analyst", "research data analyst",
        "resource strategy analyst", "retail data analyst", "right of way analyst", "risk analyst",
        "salesforce data analyst", "sap analyst", "sap business analyst",
        "sap business systems analyst", "sap hr business analyst", "sap purchasing analyst",
        "sap security system analyst", "sap systems analyst", "sap tm configuration analyst",
        "sap treasury analyst", "sccm 2007 analyst", "security assurance analyst",
        "security test analyst", "settlements analyst", "siebel crm systems analyst",
        "social listening analyst", "software qa analyst", "software quality assurance analyst",
        "sql data analyst", "sql db analyst", "sql etl analyst", "sql server bi analyst",
        "sql server data analyst", "sql server database analyst", "sql server systems analyst",
        "sr asp.net web analyst", "sr bi system analyst", "sr business analyst",
        "sr business system analyst", "sr business systems analyst", "sr data analyst",
        "sr database analyst", "sr datawarehouse analyst", "sr deployment analyst",
        "sr dotnet analyst", "sr epic beaker analyst", "sr excel / sharepoint / vb analyst",
        "sr it business analyst", "sr qa analyst", "sr risk analyst", "sr sap analyst",
        "sr sap central finance analyst", "sr sap data analyst", "sr transportation analyst",
        "sr. business analyst", "sr. business data analyst", "sr. business process analyst",
        "sr. business reporting analyst", "sr. business system analyst",
        "sr. business systems analyst", "sr. business systems analyst/project",
        "sr. cloud finops analyst", "sr. compliance analyst", "sr. data analyst",
        "sr. data analyst/data", "sr. data security analyst", "sr. data/information analyst",
        "sr. database analyst", "sr. healthcare systems integration analyst",
        "sr. information security analyst", "sr. inventory operations analyst",
        "sr. it business analyst", "sr. it security analyst", "sr. java web analyst",
        "sr. market research analyst", "sr. network analyst/architect",
        "sr. network security analyst", "sr. program analyst", "sr. risk analyst",
        "sr. salesforce crm business analyst", "sr. sap analyst", "sr. sap business systems analyst",
        "sr. sap em analyst", "sr. sap system analyst", "sr. security analyst",
        "sr. software qa analyst", "sr. sql analyst", "sr. sql database analyst",
        "sr. strategic planning analyst", "sr. system analyst", "sr. systems analyst",
        "sr. teradata bi analyst", "sr. test analyst", "sr. thermal analyst",
        "sr. use case business analyst", "sr.business analyst", "sr.systems analyst",
        "supply chain analyst", "supply chain test analyst", "support systems analyst",
        "sw data analyst/", "sw metrics analyst", "sw program analyst", "sw qa analyst",
        "sw qa automation analyst", "system analyst", "systems analyst", "systems support analyst",
        "tax analyst", "technical analyst", "technical business analyst",
        "technical marketing & project analyst", "technical support analyst",
        "techno functional analyst", "techno functional business analyst",
        "telecommunications analyst", "testing analyst", "trade compliance analyst",
        "trade techno functional analyst", "transformation ops analyst", "user experience analyst",
        "web system analyst", "windchill product analyst", "workforce planning analyst",
        "sr .net developer", "sr .net web developer", "sr android developer", "sr c++ programmer",
        "sr data engineer", "sr developer", "sr java developer", "sr python full stack developer",
        "sr software engineer", "sr. .net developer", "sr. c# programmer", "sr. c++ developer",
        "sr. c++ programmer", "sr. developer", "sr. full stack engineer/architect",
        "sr. full stack engineer/architect1", "sr. fullstack developer", "sr. fullstack engineer",
        "sr. java developer", "sr. java programmer", "sr. net developer", "sr. programmer",
        "sr. python developer", "sr. software engineer", "sr .net back end developer",
        "sr .net sw engineer", "sr .net web applications developer",
        "sr 3d graphics pipeline engineer", "sr aem developer", "sr android application developer",
        "sr application developer", "sr automation qa engineer", "sr automation validation engineer",
        "sr backend software engineer", "sr bi cube developer", "sr bi developer",
        "sr big data engineer", "sr build & release engineer", "sr build and release engineer",
        "sr c#.net developer", "sr c++ linux developer", "sr c/c++ developer",
        "sr c/c++ firmware developer", "sr c/c++software engineer", "sr cad engineer",
        "sr cold fusion developer", "sr data analytics developer", "sr design engineer",
        "sr hybris developer", "sr ibm websphere commerce developer", "sr informatica etl developer",
        "sr java engineer", "sr mainframe developer", "sr maximo developer", "sr network engineer",
        "sr oracle apex developer", "sr oracle data developer", "sr qa engineer",
        "sr rtl design engineer", "sr signal integrity engineer", "sr soc hardware design engineer",
        "sr software developer", "sr ssis developer", "sr sw automation developer",
        "sr sw web developer", "sr system protection engineer", "sr system sw engineer",
        "sr systems integration engineer", "sr test automation engineer", "sr test engineer",
        "sr thermal mechanical engineer", "sr veritification engineer", "sr web ui developer",
        "sr widows driver developer", "sr win8/metro ui apps developer",
        "sr. .net angular developer", "sr. .net web developer", "sr. .net web integration developer",
        "sr. android developer", "sr. application developer", "sr. applications developer",
        "sr. applications engineer", "sr. asic automation design engineer",
        "sr. asic design engineer", "sr. asp.net and sql server developer", "sr. asp.net programmer",
        "sr. asp.net web developer", "sr. automation test engineer", "sr. aws api developer",
        "sr. aws cloud applications developer", "sr. back end software engineer",
        "sr. backend developer", "sr. backend software engineer", "sr. big data engineer",
        "sr. bios firm ware engineer", "sr. business process engineer", "sr. c# .net developer",
        "sr. c# software engineer", "sr. c# web developer", "sr. c#, .net developer",
        "sr. c#.net developer", "sr. c#.net web applications developer", "sr. c#.net web developer",
        "sr. c++ software engineer", "sr. c++ sw engineer", "sr. c++ validation engineer",
        "sr. c++, asp.net sw engineer", "sr. c/ c++ developer", "sr. c/c++ developer",
        "sr. c/c++ sw developer", "sr. cad methods engineer", "sr. cad support engineer",
        "sr. catia methods engineer", "sr. certified filemaker developer",
        "sr. cobol mainframe programmer", "sr. cognos bi reports developer", "sr. cognos developer",
        "sr. cognos reports developer", "sr. coldfusion developer", "sr. data analytics engineer",
        "sr. data engineer", "sr. data engineer,", "sr. data visualization developer",
        "sr. database developer", "sr. device driver developer", "sr. devops engineer",
        "sr. documentum developer", "sr. dot net developer", "sr. driver developer",
        "sr. drupal developer", "sr. drupal web applications developer",
        "sr. embedded linux developer", "sr. embedded sw engineer",
        "sr. enterprise sw applications engineer", "sr. etl test engineer",
        "sr. filemaker developer", "sr. firmware engineer", "sr. front end developer",
        "sr. front end java ui developer", "sr. front end ui developer", "sr. frontend developer",
        "sr. full stack aws developer", "sr. full stack software developer",
        "sr. fullstack web developer", "sr. hardware design engineer", "sr. hybris developer",
        "sr. informatica analytics developer", "sr. information security engineer",
        "sr. integration and support engineer", "sr. ios developer",
        "sr. j2ee cloud security applications developer", "sr. j2ee web developer",
        "sr. java application developer", "sr. java full stack developer", "sr. java ui developer",
        "sr. java web developer", "sr. lamp applications developer", "sr. linux developer",
        "sr. linux kernel c++ systems engineer", "sr. linux systems engineer",
        "sr. machine learning algorithm developer", "sr. mainframe bi programmer",
        "sr. mainframe developer", "sr. media flash validation engineer", "sr. ml engineer",
        "sr. mobile application developer", "sr. mobile applications developer",
        "sr. mobile developer", "sr. multimedia test developer", "sr. network engineer",
        "sr. network security engineer", "sr. oracle c2m integration developer",
        "sr. oracle developer", "sr. oracle forms developer", "sr. php web developer",
        "sr. pl/sql oracle developer", "sr. power bi developer", "sr. powerbuilder developer",
        "sr. powerbuilder programmer", "sr. python application developer",
        "sr. qa automation/performance test engineer", "sr. quality engineer",
        "sr. rtl design engineer", "sr. ruby web services developer", "sr. salesforce developer",
        "sr. sap data services developer", "sr. sap developer", "sr. sharepoint bi developer",
        "sr. sharepoint developer.", "sr. site reliability engineer",
        "sr. software configuration engineer", "sr. software developer", "sr. software qa engineer",
        "sr. software web developer", "sr. sql bi developer", "sr. sql developer",
        "sr. sql server report developer", "sr. sql sw developer", "sr. sw developer",
        "sr. system engineer", "sr. system software engineer", "sr. systems integration engineer",
        "sr. tcl systems programmer", "sr. validation engineer",
        "sr. vb6 and crystal reports developer", "sr. vb6 programmer",
        "sr. virtualization developer", "sr. visual basic 6.0 programmer",
        "sr. web applications developer", "sr. web developer", "sr. web software developer",
        "sr. windows driver developer", "sr. windows installer c++ developer",
        "sr. wireless automation developer", "sr. xml developer", "sr.. net developer",
        "sr.android software engineer", "sr.application developer", "sr.asp.net developer",
        "sr.build engineer", "sr.c++ validation engineer", "sr.c/c++ network evaluation engineer",
        "sr.cad engineer", "sr.excel vba software developer", "sr.firmware engineer",
        "sr.graphics software engineer", "sr.human factors engineer", "sr.linux validation engineer",
        "sr.ms visual studio sw developer", "sr.net developer", "sr.net. developer",
        "sr.network engineer", "sr.network validation engineer",
        "sr.pcie gen3 debug and validation engineer", "sr.qa engineer", "sr.qa test engineer",
        "sr.scm engineer", "sr.sitecore developer", "sr.sle validation engineer",
        "sr.software application engineer", "sr.software developer", "sr.software engineer",
        "sr.software programmer", "sr.sw developer", "sr.system engineer", "sr.systems engineer",
        "sr.validation engineer", "sr.verification engineer", "sr.web developer",
        "sr.win8 apps developer", "sr.windows build engineer",
        "3d pipeline engineer", "Network Engineer", "QA Engineer", "Security Engineer",
        "Software Engineer", "ab initio developer", "ab initio etl developer",
        "adroid/linux sw developer", "analog design engineer", "analog engineer",
        "analog io validation engineer", "analytical applications and tool developer",
        "analytical software engineer", "android app developer", "android application developer",
        "android automation engineer", "android developer", "android device driver developer",
        "android device drivers developer", "android driver developer",
        "android driver validation engineer", "android middleware developer",
        "android mobile automation test engineer", "android os / system developer",
        "android qa engineer", "android sdk developer", "android software engineer",
        "android sw drivers developer", "android sw engineer", "android sw test engineer",
        "android systems developer", "android systems integration/scm engineer",
        "android test developer", "android validation engineer", "android/ios test developer",
        "angular developer", "angular js and node.js developer", "api & web developer",
        "api developer", "api test automation engineer", "application developer",
        "application developer/programmer", "application engineer", "application packaging engineer",
        "application programmer", "application release engineer", "application sw developer",
        "applications developer", "applications engineer", "applications programmer",
        "applications support engineer.", "applications systems engineer", "arc gis developer",
        "arcgis developer", "arcgis web application developer", "asic design engineer",
        "asic logic verification engineer", "asic physical design automation engineer",
        "asic physical design engineer", "asic verification engineer", "asp. net sw engineer",
        "asp.net database developer", "asp.net developer", "asp.net programmer",
        "asp.net web applications developer", "asp.net web developer", "asp.net web ui developer",
        "audio driver validation engineer", "automated test equipment validation engineer",
        "automation engineer", "automation plc engineer", "automation qa engineer",
        "automation sw qa engineer", "automation test design engineer", "automation test developer",
        "automation test engineer", "aws & sysops or devops engineer", "aws application engineer",
        "aws cloud developer", "aws cloud engineer", "aws connect engineer", "aws data engineer",
        "aws developer", "aws devops engineer", "azure engineer", "back end web developer",
        "backend api developer", "backend drupal developer", "backend java developer",
        "backend software engineer", "backend web developer", "bi developer", "bi qa engineer",
        "bi reports developer", "big data engineer", "bios test engineer", "bo developer",
        "board design engineer", "bobj applications developer", "build and integration engineer",
        "build and release engineer", "build and release systems support engineer",
        "build automation engineer", "build engineer", "build engineer/systems",
        "build release engineer/developer", "build sw engineer", "business intelligence developer",
        "business objects developer", "business process engineer", "bw4/hana developer",
        "c / c++ android software engineer", "c / c++ sw engineer", "c /c++ developer",
        "c# .net applications engineer", "c# .net developer", "c# .net software programmer",
        "c# .net web developer", "c# / .net developer", "c# and wpf developer",
        "c# application developer", "c# asp.net developer", "c# desktop application developer",
        "c# desktop application programmer", "c# developer", "c# microsoft developer",
        "c# programmer", "c# web developer", "c# window application developer",
        "c# windows application developer", "c#. sw developer", "c#.net application programmer",
        "c#.net application ui developer", "c#.net applications and web developer",
        "c#.net applications programmer", "c#.net desktop applications programmer",
        "c#.net developer", "c#.net programmer", "c#.net software engineer", "c#.net sw engineer",
        "c#.net ui developer", "c#.net web application developer", "c#.net web developer",
        "c#.net web services developer", "c#.net web ui developer",
        "c#.net windows applications developer", "c#/ .net programmer", "c#asp.net developer",
        "c#asp.net programmer", "c#asp.net web developer", "c++ developer", "c++ driver developer",
        "c++ embedded software engineer", "c++ embedded sw engineer", "c++ firmware developer",
        "c++ graphics validation engineer", "c++ gui developer", "c++ linux developer",
        "c++ middleware developer", "c++ programmer", "c++ software developer",
        "c++ software engineer", "c++ sw developer", "c++ sw development engineer",
        "c++ sw engineer", "c++ systems debug engineer", "c++ test automation developer",
        "c++ test automation engineer", "c++ validation engineer", "c++ windows developer",
        "c++ windows programmer", "c++/c# sw engineer", "c++/java android developer",
        "c++/java developer", "c++graphics software engineer", "c, c++ developer",
        "c, c++ software developer", "c, c++ software engineer", "c, c++design automation engineer",
        "c, c++sw developer", "c/c++ application engineer", "c/c++ cloud systems engineer",
        "c/c++ developer", "c/c++ programmer", "c/c++ software developer", "c/c++ software engineer",
        "c/c++ sw developer", "c/c++ sw engineer", "c/c++ sw programmer", "c/c++ sw test developer",
        "c/c++ systems programmer", "c/c++ validation engineer", "c/c++embedded sw engineer",
        "c/c++sw developer", "c/c++validation engineer", "ca gen applications developer",
        "cad design automation engineer", "cad design engineer", "cad engineer",
        "cad program engineer", "cadence pcb layout design engineer",
        "capital harness support engineer", "catia v5 support engineer",
        "certified network security engineer", "chip validation engineer", "ci automation engineer",
        "circuit board validation engineer", "circuit design engineer",
        "circuit layout design engineer", "cisco network engineer", "cisco voip engineer",
        "cloud developer", "cloud engineer", "cloud infrastructure engineer",
        "cloud orchestration engineer", "cloud programmer", "cloud sw developer",
        "cloud systems engineer", "cloudera developer", "cms systems support engineer",
        "cobal developer", "cobol developer", "cobol programmer", "cobol software engineer",
        "cobol/ db2 application developer", "cobol/db2 application developer",
        "cognos report developer", "cold fusion developer", "coldfusion / java developer",
        "coldfusion developer", "coldfusion programmer", "coldfusion web applications developer",
        "coldfusion web developer", "componenet design engineer", "component design engineer",
        "computer / sw engineer", "computer programmer", "computer systems engineer",
        "contact center solutions engineer", "csd sw security developer",
        "customer support engineer", "cybersecurity engineer", "data base developer",
        "data engineer", "data platform engineer", "data visualization developer",
        "data warehouse developer", "database developer", "database programmer",
        "database test engineer", "datastage developer", "datastage etl developer",
        "ddr/dimm validation engineer", "design automation engineer", "design engineer",
        "dev ops engineer", "developer", "device driver developer",
        "device driver validation engineer", "device simulation developer",
        "devops application engineer", "devops developer", "devops engineer", "devsecops engineer",
        "digital product roadmap developer", "drupal developer", "drupal web developer",
        "ecryption coding engineer", "eda engineer", "edi developer", "electric engineer",
        "electrical design engineer", "electrical engineer", "electrical validation engineer",
        "electronic design automation engineer", "electronics and communications engineer",
        "embedded android software developer", "embedded design engineer", "embedded developer",
        "embedded device driver sw developer", "embedded device sw developer",
        "embedded drivers debug engineer", "embedded engineer", "embedded firmware developer",
        "embedded linux engineer", "embedded software and firmware developer",
        "embedded software developer", "embedded software engineer", "embedded sw developer",
        "embedded sw engineer", "embedded systems developer", "embedded systems engineer",
        "embedded systems programmer", "embedded windows developer", "embedded wireless programmer",
        "engineer", "engineer:", "enterprise design systems engineer", "environmental engineer",
        "erlang programmer", "esri/javascript api developer", "etl data developer",
        "etl data warehouse engineer", "etl developer", "excel programmer", "excel vba developer",
        "fc rtl engineer", "firmware developer", "firmware engineer", "firmware linux developer",
        "firmware software engineer", "firmware sw developer", "firmware sw validation engineer",
        "firmware test engineer", "firmware testing engineer", "firmware validation engineer",
        "fpga design engineer", "fpga developer", "fpga engineer", "fpga hardware engineer",
        "fpga validation engineer", "fpga verification engineer", "front end developer",
        "front end java developer", "front end react developer", "front end web developer",
        "full stack .net developer", "full stack application, database and etl developer",
        "full stack back end developer", "full stack developer", "full stack engineer",
        "full stack java developer", "full web stack developer", "fullstack .net developer",
        "fullstack engineer", "fw/sw dev storage test engineer", "gas pipeline engineer",
        "git systems engineer", "graph ql developer/react", "graphic validation engineer",
        "graphics automation engineer", "graphics design engineer",
        "graphics driver validation engineer", "graphics hardware engineer",
        "graphics software engineer", "graphics validation engineer", "graphql api developer",
        "gui design engineer", "gui developer", "hadoop bigdata engineer", "hadoop developer",
        "hardware component design engineer", "hardware design engineer", "hardware engineer",
        "hardware test engineer", "help desk support engineer", "hpc parallel programmer",
        "human factors engineer", "hw design engineer", "hw engineer", "hw mask design engineer",
        "hw mechanical design engineer", "hw sw validation engineer",
        "hw systems integration engineer", "hw test automation engineer", "hw test engineer",
        "hw thermal mechanical engineer", "hw validation engineer", "hw verification engineer",
        "hw/sw validation engineer", "hybris developer", "hybris ecommerce developer",
        "hybris ui developer", "iam engineer", "iam identityiq developer", "iam qa engineer",
        "ibm bpm developer", "ibm datapower iid/wid integration developer",
        "ibm datastage / datapower developer", "ibm datastage developer",
        "ibm filenet applications engineer", "ibm lotus notes developer", "ibm middleware developer",
        "ibm websphere datapower developer", "ibm websphere developer",
        "ic cad reliability engineer", "ic design automation engineer", "ic design engineer",
        "ic layout design engineer", "ic verfication engineer", "ic verification engineer",
        "image quality test engineer", "image quality testing engineer", "image test engineer",
        "infinium programmer", "informatica developer", "information security engineer",
        "infotainment engineer", "infrastructure application engineer", "infrastructure engineer",
        "infrastructure qa test engineer", "infrastructure systems engineer",
        "infrastructure test engineer", "infrastructure validation engineer",
        "installshield developer", "intermediate cognos developer",
        "intermediate java web developer", "internal data engineer", "internal devops engineer",
        "internet software engineer", "ios and android developer", "ios applications developer",
        "ios developer", "ios mobile app developer", "ios mobile automation test engineer",
        "ios software developer", "ios software engineer", "ios sw developer",
        "ip software/fpga engineer", "it integration engineer", "it reports developer",
        "it support engineer", "it systems engineer", "it systems support engineer",
        "it technical marketing engineer", "j2ee applications developer", "j2ee developer",
        "j2ee java web developer", "j2ee sw engineer", "j2ee web 2.0 developer",
        "j2ee web developer", "jaspersoft developer", "java api programmer",
        "java application developer", "java applications developer", "java applications engineer",
        "java developer", "java edi developer", "java engineer", "java full stack developer",
        "java fullstack developer", "java gui developer", "java programmer",
        "java software engineer", "java sw developer", "java web applications developer",
        "java web developer", "java web services developer", "javascript & node.js developer",
        "javascript web developer", "jd edwards developer", "jira developer",
        "jr test automation engineer", "jr. .net developer", "jr. c#.net programmer",
        "jr. firmware engineer", "jr. python developer", "jr. win device drivers developer",
        "jr. wireless mobile developer", "jr.excel vba developer", "jython developer",
        "kubernetes platform engineer", "lab engineer", "labview engineer",
        "lawson financial programmer", "linux application developer",
        "linux build and release engineer", "linux c++ sw engineer", "linux driver developer",
        "linux kernel developer", "linux kernel validation engineer",
        "linux network driver developer", "linux network engineer",
        "linux networking development engineer", "linux programmer", "linux qa engineer",
        "linux software debug engineer", "linux software developer", "linux software engineer",
        "linux sw developer", "linux sw engineer", "linux systems engineer",
        "linux systems test automation engineer", "linux systems validation engineer",
        "linux test automation engineer", "linux test engineer", "linux validation engineer",
        "linux virtualization support engineer", "logic design engineer;",
        "logic design verification engineer", "lotus notes domino developer",
        "m2m software engineer", "mac developer", "machine learning application developer",
        "machine learning developer", "machine learning engineer", "magento developer",
        "mainframe developer", "mainframe programmer", "mainframe support engineer",
        "mainframes test engineer", "manufacturing engineer", "mask design engineer",
        "matlab developer", "matlab software engineer", "maximo application developer",
        "maximo developer", "mbuild engineer", "mechanical design engineer", "mechanical engineer",
        "media software developer", "microsoft access programmer",
        "microsoft kernel validation engineer", "microsoft systems engineer",
        "microstrategy report developer", "mid level browser developer",
        "mid level c/c++ software device driver developer", "middleware engineer",
        "middleware software engineer", "ml data engineer",
        "mobile android camera driver and application developer", "mobile application developer",
        "mobile applications developer", "mobile automation test engineer", "mobile developer",
        "mobile firmware validation engineer", "mobile front end web developer",
        "mobile linux qa test developer", "mobile qa engineer", "mobile sw application developer",
        "mobile systems validation engineer", "mobile website sw developer",
        "ms dynamics 365 developer", "ms exchange security engineer", "ms power platform developer",
        "ms sql bi developer", "ms sql database developer", "ms sql developer",
        "ms sql reporting services developer", "mule soft developer", "mulesoft developer",
        "multimedia test engineer", "net application programmer", "net applications programmer",
        "net developer", "net full stack developer", "net fullstack developer",
        "net mobile application developer", "net programmer", "net qa engineer",
        "net software engineer", "net sw applications developer", "net web application developer",
        "net web applications developer", "net web developer", "net web programmer",
        "net web services developer", "net web ui developer", "net webservices developer",
        "net windows developer", "net, c# sw developer", "network driver test engineer",
        "network engineer", "network evaluation engineer", "network infrastructure engineer",
        "network security engineer", "network software engineer",
        "network software validation engineer", "network software/hardware validation engineer",
        "network systems engineer", "network systems programmer", "network systems sw developer",
        "network test engineer", "network validation engineer", "nextiva system engineer",
        "obiee developer", "online analytical processing developer", "open source web developer",
        "opengl developer", "opentext exstream developer", "operating systems programmer",
        "operations technology systems engineer", "oracle apex developer",
        "oracle apex pos developer", "oracle application developer",
        "oracle application express developer", "oracle apps test engineer", "oracle bi developer",
        "oracle cc&b / mdm developer", "oracle database engineer", "oracle integration engineer",
        "oracle odi developer", "oracle oim developer", "oracle pl/sql developer",
        "oracle programmer", "oracle reports developer", "oracle seibel bi developer",
        "oracle soa integration developer", "oracle soa suite developer",
        "oracles' application express developer", "os modem software engineer",
        "os x driver developer", "osx developer", "osx sw developer", "outsystems developer",
        "pc application test engineer", "pc validation engineer", "pcb cad design engineer",
        "pcb cad engineer", "pcb cad layout design engineer", "pcb design engineer",
        "pcb layout design engineer", "pcb layout engineer",
        "pcie gen3 debug and validation engineer", "peoplesoft developer",
        "peoplesoft financials developer", "peoplesoft financials technical developer",
        "peoplesoft hcm developer", "performance test engineer", "perl application developer",
        "php developer", "php laravel developer", "php web developer", "physical design engineer",
        "pl/sql developer", "pl/sql oracle developer", "platform software engineer",
        "platform validation engineer", "playready drm engineer", "plsql programmer",
        "position: programmer", "postgre sql dba/programmer", "power apps developer",
        "power bi developer", "power platform developer", "power validation engineer",
        "powerbuilder architect/developer", "powerbuilder developer",
        "pro/e mechanical design engineer", "product developer", "product development engineer",
        "product engineer", "product localization engineer", "product test engineer",
        "proe mechanical engineer", "programmer", "progressive mobile applications developer",
        "project engineer", "purview security engineer", "python api developer",
        "python application developer", "python developer", "python developers", "python engineer",
        "python web developer", "qa automation engineer", "qa engineer",
        "qa test automation engineer", "qa test engineer", "qa testing engineer",
        "quality assurance engineer", "quality engineer", "qualys patch engineer",
        "quickbooks developer", "r&d engineer", "raid storage validation design engineer",
        "react developer", "relational database developer", "release engineer",
        "reliability design automation engineer", "reliability engineer", "report developer",
        "reports developer", "rf design engineer", "rpa developer", "rtl component design engineer",
        "rtl design automation engineer", "rtl design engineer", "rtl hw design engineer",
        "rtl sw engineer", "rtl validation engineer", "rtl verification engineer",
        "ruby on rails developer", "ruby on rails engineer", "ruby on rails web developer",
        "s/w quality assurance engineer", "salesforce administrator/developer",
        "salesforce developer", "salesforce engineer", "san validation engineer",
        "sap application engineer", "sap bo bi developer", "sap business object report developer",
        "sap bw developer", "sap bw4/hana expert engineer", "sap developer", "sap mdg developer",
        "sap mm test engineer", "sap performance test engineer", "scala applications developer",
        "section 508 test engineer", "security development engineer", "security engineer",
        "server firmware developer", "servicenow application developer", "servicenow developer",
        "sharepoint 2010 developer", "sharepoint 2013 developer", "sharepoint application developer",
        "sharepoint bi developer", "sharepoint developer", "sharepoint intranet developer",
        "sharepoint migration developer", "sharepoint solutions developer",
        "sharepoint web developer", "si validation engineer", "signal integrity engineer",
        "signal processing engineer", "silicon design automation engineer", "simulation engineer",
        "site reliability engineer", "smalltalk developer", "smart phone app developer",
        "smartphones validation engineer", "soc design engineer", "soc design verification engineer",
        "soc validation engineer", "socket validation engineer", "software application developer",
        "software applications developer", "software applications engineer",
        "software build and release engineer", "software build engineer",
        "software build release engineer", "software developer", "software engineer",
        "software qa engineer", "software systems support engineer",
        "software test development engineer", "software test engineer",
        "software validation engineer", "software validation test engineer",
        "software/electrical engineer", "splunk developer", "sql application developer",
        "sql bi architect/developer", "sql bi report developer", "sql database developer",
        "sql developer", "sql programmer", "sql server bi application developer",
        "sql server bi developer", "sql server bi engineer", "sql server db developer",
        "sql server developer", "sql web developer", "ssd firmware engineer",
        "ssd systems engineer", "storage engineer", "stress hardware development engineer",
        "structural design engineer", "sw /hw test developer", "sw automation engineer",
        "sw automation qa engineer", "sw automation test engineer", "sw backend web developer",
        "sw build & release engineer", "sw build automation engineer", "sw build engineer",
        "sw database developer", "sw design engineer", "sw developer", "sw development engineer",
        "sw development quality engineer", "sw device driver developer", "sw engineer",
        "sw firmware developer", "sw infrastructure engineer", "sw localization test engineer",
        "sw product qa engineer", "sw programmer", "sw qa automation engineer", "sw qa engineer",
        "sw qa tester engineer", "sw quality engineer", "sw release engineer",
        "sw systems application developer", "sw systems engineer",
        "sw test automation development engineer", "sw test automation engineer", "sw test engineer",
        "sw validation engineer", "sw validation tools developer", "sw/fw developer",
        "sw/fw integration engineer", "sw/fw validation engineer", "sybase powerbuilder developer",
        "synopsys validation engineer", "system engineer", "system verilog logic design engineer",
        "systems design engineer", "systems engineer", "systems integration engineer",
        "systems programmer", "systems software developer", "systems software engineer",
        "systems test engineer", "systems validation engineer", "systems/ software engineer",
        "syteline erp application developer", "tableau bi developer", "tableau developer",
        "tableau reports developer", "tech writer/web developer/tester",
        "technical marketing engineer", "technical support engineer", "telecom engineer",
        "telecommunications engineer", "test automation developer", "test automation engineer",
        "test automation tools engineer", "test bench developer", "test design automation engineer",
        "test engineer", "test execution engineer", "testbench developer",
        "thermal mechanical engineer", "tivoli developer", "tool component design engineer",
        "typescript developer", "uat test engineer", "ui developer", "ui engineer",
        "ui/ux developer", "user productivity kit content developer", "ux design and react engineer",
        "ux design engineer", "ux developer", "validation automation engineer",
        "validation engineer", "validation test automation engineer",
        "validation test design engineer", "vb .net developer", "vb developer", "vb.net developer",
        "vb.net programmer", "vb.net web developer", "vb6, coldfusion programmer", "vba developer",
        "verification engineer", "verilog design engineer", "verilog design verification engineer",
        "verilog validation engineer", "verilog verification engineer", "video coding engineer",
        "virtual networks engineer", "visual basic 6 programmer", "visual c# developer",
        "visual studio sw developer", "voip engineer", "wcf/azure developer", "web app developer",
        "web application developer", "web applications developer",
        "web backend application developer", "web developer", "web qa engineer",
        "web services developer", "web ui developer", "webmethods developer", "website developer",
        "websphere commerce developer", "websphere integration engineer",
        "windows .net c# developer", "windows 8 systems programmer", "windows application developer",
        "windows applications systems engineer", "windows client app developer", "windows developer",
        "windows driver developer", "windows drivers developer",
        "windows drivers validation engineer", "windows installer developer",
        "windows middleware developer", "windows support engineer", "windows sw developer",
        "windows sys debug developer", "windows systems engineer", "windows systems programmer",
        "windows systems test engineer", "windows/linux systems engineer",
        "wired ethernet validation engineer", "wireless network engineer",
        "wireless systems engineer", "wix developer", "xcelsius developer"
}

# Extra tokens that should cause rejection if present in the candidate line
NON_TITLE_INDICATORS = [
    'project', 'project title', 'portfolio', 'client', 'application', 'including',
    'responsibilities', 'technologies', 'duration', 'module', 'feature', 'functionality',
    'description', 'achievement', 'location', 'location :', 'certification',
    'interpersonal', 'skills', 'communication', 'teamwork', 'problem-solving',
    'institute', 'institution', 'university', 'college', 'school', 'board', 'education',
    'b.tech', 'btech', 'm.tech', 'mtech', 'b.e', 'be', 'm.e', 'me', 
    'bachelor', 'master', 'degree', 'qualification', 'bsc', 'msc', 'bca', 'mca', 'ba', 'ma'
]

# Soft skills that are NOT job titles (should be deprioritized or rejected)
SOFT_SKILLS = [
    'leadership', 'communication', 'teamwork', 'collaboration', 'problem-solving',
    'analytical', 'creative', 'detail-oriented', 'self-motivated', 'adaptable'
]

# Verbs that indicate descriptive sentence (not a title)
DESCRIPTION_VERBS = [
    'developing', 'designing', 'implementing', 'managing', 'working', 'led', 'lead',
    'ensuring', 'responsible', 'responsibilities', 'implemented', 'developed'
]

# Minimum fuzzy ratio to consider a match good
FUZZY_THRESHOLD = 0.78


# -------------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    t = text.replace('\r\n', '\n').replace('\r', '\n')
    # Normalize repeated whitespace and strip
    t = re.sub(r'\t+', ' ', t)
    t = re.sub(r'[ \u00A0]+', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()


def title_case(s: str) -> str:
    # keep common uppercase words intact, but simple titlecasing is fine for designations
    return ' '.join([w.capitalize() if len(w) > 1 else w.upper() for w in s.split()])


def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def split_title_company(line: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Split a line that might be in "Title - Company" or "Title – Company" format.
    Returns (title_part, company_part) or (None, None) if not in this format.
    Handles regular hyphen (-), en-dash (–), and em-dash (—).
    """
    # Try different dash types: regular hyphen, en-dash, em-dash
    for dash in [' - ', ' – ', ' — ', ' -', ' –', ' —']:
        if dash in line:
            parts = line.split(dash, 1)
            if len(parts) == 2:
                return (parts[0].strip(), parts[1].strip())
    return (None, None)


def has_dash_separator(line: str) -> bool:
    """
    Check if line contains a dash separator (hyphen, en-dash, or em-dash).
    """
    return ' - ' in line or ' – ' in line or ' — ' in line or line.startswith((' -', ' –', ' —'))


# -------------------------------------------------------------------------
# CORE: extracting a cleaned "experience" segment (to avoid project sections)
# -------------------------------------------------------------------------
def crop_to_experience_section(text: str) -> str:
    """
    Try to return the portion of text that is the experience / work history.
    If no explicit header, return the first 2500 chars but stop at portfolio/project section.
    """
    lower = text.lower()
    # find experience-like headers
    m = re.search(r'(experience|work history|employment history|professional experience|career development)', lower)
    if m:
        start = m.start()
        slice_text = text[start:]
    else:
        # fallback: start from top (many resumes don't have headers)
        slice_text = text[:4000]

    # Cut off at portfolio/project section to avoid project-title becoming job title
    stop = re.search(r'(portfolio of projects|portfolio|project\s+\d+|project title|projects|project:)', slice_text, re.IGNORECASE)
    if stop:
        slice_text = slice_text[:stop.start()]
    
    # CRITICAL: Cut off at certifications section to avoid certification titles being mistaken for job titles
    # This prevents "Salesforce Developer" from certifications being extracted as current designation
    cert_stop = re.search(r'\b(certifications?|certificate|certified|cert\.?)\s*:?\s*$', slice_text, re.IGNORECASE | re.MULTILINE)
    if cert_stop:
        slice_text = slice_text[:cert_stop.start()]
    
    # CRITICAL: Cut off at education section to avoid institution names being mistaken for job titles
    # This prevents "Gates Institute Of Technology" or "Director Of Technology" from education being extracted
    # Also prevents matching "Director" from "Director Of Technology" in education section
    edu_stop = re.search(r'\b(education|educational|academic|qualification|degree|b\.?tech|m\.?tech|b\.?e\.?|m\.?e\.?|bachelor|master|ph\.?d|doctorate)\s*:?\s*$', slice_text, re.IGNORECASE | re.MULTILINE)
    if edu_stop:
        slice_text = slice_text[:edu_stop.start()]

    return slice_text.strip()


# -------------------------------------------------------------------------
# Candidate line detection: select lines that look like job title lines
# -------------------------------------------------------------------------
def candidate_title_lines(segment: str) -> List[Tuple[int, str]]:
    """
    Return list of (score_priority, line) where higher priority lines should be checked first.
    We look for lines that contain seniority keywords or end with common job role suffixes.
    """
    lines = []
    for raw in segment.split('\n'):
        line = raw.strip()
        if not line:
            continue
        # Skip very long lines (unlikely to be titles)
        if len(line) > 200:
            continue
        # drop obvious contact / email / phone lines
        if re.search(r'(@|\bemail\b|\bphone\b|\+?\d{5,})', line, re.IGNORECASE):
            continue
        # Skip lines that start with bullet points or dashes (likely descriptions)
        if re.match(r'^[\-\–\—\•\*\u2022]\s+', line):
            continue
        # Skip lines that start with action verbs (likely descriptions)
        if re.match(r'^(playing|managing|strategizing|front|optimizing|empowering|coaching|initiated|standardizing|formalizing|responsible|establishing|improving|has|utilized|developed|created|configured|implemented|enabled|designed|integrated|automated|conducted|deployed|provided|offered|achieved|enhanced|reduced|improved|streamlined|optimized)\s+', line, re.IGNORECASE):
            continue
        # Skip lines that start with "in" + verb (sentence fragments like "in addressing")
        if re.match(r'^in\s+(addressing|managing|providing|ensuring|developing|creating|implementing|designing|building|establishing|leading|coordinating|facilitating|supporting|maintaining|improving|optimizing|streamlining|delivering|executing|performing|conducting|handling|overseeing|supervising|directing|guiding|assisting|helping|contributing|participating|involving|working|serving|acting|functioning|operating|running|controlling|managing)\s+', line, re.IGNORECASE):
            continue
        # Skip lines that are clearly sentence fragments (start with lowercase, contain "R&D", etc.)
        if re.match(r'^[a-z]', line) and not line[0].isupper():
            # But allow if it's a known title that starts lowercase
            if not any(title in line.lower() for title in ['engineer', 'developer', 'consultant', 'manager', 'director', 'analyst', 'architect']):
                continue
        lines.append(line)

    # First, identify lines with "till date", "present", "current" patterns for priority boost
    current_date_patterns = re.compile(r'\b(till\s+date|to\s+date|present|current|till\s+now|ongoing)\b', re.IGNORECASE)
    current_date_range_pattern = re.compile(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|[A-Za-z]+)\s+\d{4}\s+(?:to|till|-)\s+(?:till\s+date|date|present|current|till\s+now|ongoing)\b', re.IGNORECASE)
    
    # Split segment into lines to check proximity
    segment_lines = segment.split('\n')
    
    prioritized = []
    # Give higher weight to lines that contain seniority or core role keywords
    for line_idx, line in enumerate(lines):
        lw = line.lower()
        score = 0
        
        # MAJOR BOOST: Check if this line is near a "till date" or "present" date
        # Find the line in the segment
        for seg_idx, seg_line in enumerate(segment_lines):
            if line in seg_line or seg_line.strip() == line.strip():
                # Check nearby lines (within 3 lines) for "till date" patterns
                for check_idx in range(max(0, seg_idx - 3), min(len(segment_lines), seg_idx + 4)):
                    check_line = segment_lines[check_idx].lower()
                    # Check for explicit date range with "till date"
                    if current_date_range_pattern.search(segment_lines[check_idx]):
                        score += 30  # Major boost for explicit current date ranges
                        break
                    # Check for "till date" or "present" keywords nearby
                    elif current_date_patterns.search(check_line):
                        score += 20  # Boost for current position indicators
                        break
                break
        
        # Check if it's a soft skill (should be heavily deprioritized)
        if any(skill in lw for skill in SOFT_SKILLS):
            score -= 10  # Heavy penalty for soft skills
        
        # Check for actual job title keywords (prioritize these)
        title_keywords = ['senior', 'sr', 'manager', 'director', 'consultant', 'engineer', 'developer', 'architect', 'analyst', 'lead', 'principal']
        if any(keyword in lw for keyword in title_keywords):
            # But only if it's part of a real title pattern, not just "leadership"
            if lw in SOFT_SKILLS:
                score -= 5  # Penalty for soft skills that contain keywords
            else:
                score += 3
        
        # lines that are short (<=6 words) are more likely to be titles
        if len(lw.split()) <= 6:
            score += 2
        
        # Bonus for lines that match known title patterns exactly
        if any(title in lw for title in ['senior consultant', 'director', 'software engineer', 'technical director']):
            score += 5
        
        # if the line contains a dash followed by words (e.g., "Company - Senior Consultant"), lower priority
        # Check for all dash types: hyphen, en-dash, em-dash
        if re.search(r'\s[-–—]\s', line):
            score -= 1
        # deprioritize if it looks like "Company, Location"
        if re.match(r'^[A-Za-z0-9 &.,-]+,\s*[A-Za-z ]+$', line):
            score -= 2
        
        # Penalty for lines that are just single words (unless they're known titles)
        if len(lw.split()) == 1 and lw not in ['engineer', 'developer', 'consultant', 'manager', 'director', 'analyst', 'architect']:
            score -= 3
            
        prioritized.append((score, line))
    prioritized.sort(key=lambda x: x[0], reverse=True)
    return prioritized


# -------------------------------------------------------------------------
# Heuristics / Matching
# -------------------------------------------------------------------------
def is_invalid_title_line(line: str) -> bool:
    lw = line.lower().strip()
    
    # Reject soft skills that are not job titles
    if lw in SOFT_SKILLS:
        return True
    
    # Reject lines that start with "in" + verb (sentence fragments)
    if re.match(r'^in\s+(addressing|managing|providing|ensuring|developing|creating|implementing|designing|building|establishing|leading|coordinating|facilitating|supporting|maintaining|improving|optimizing|streamlining|delivering|executing|performing|conducting|handling|overseeing|supervising|directing|guiding|assisting|helping|contributing|participating|involving|working|serving|acting|functioning|operating|running|controlling|managing)\s+', lw):
        return True
    
    # Reject lines containing "R&D" or "R & D" (research and development - not a title)
    if re.search(r'\br\s*[&]\s*d\b', lw, re.IGNORECASE):
        return True
    
    # CRITICAL: Reject lines that are clearly education/institution names
    # This prevents "Gates Institute Of Technology" or "Director Of Technology" from education being extracted
    education_keywords = ['institute', 'institution', 'university', 'college', 'school', 'board', 
                         'b.tech', 'btech', 'm.tech', 'mtech', 'b.e', 'be', 'm.e', 'me', 
                         'bachelor', 'master', 'degree', 'qualification', 'education', 'academic',
                         'bsc', 'msc', 'bca', 'mca', 'ba', 'ma', 'phd', 'doctorate']
    
    # Check for degree patterns (e.g., "Btech, Civil Engineering", "B.Tech (CSE)", "Bachelor of Engineering")
    degree_patterns = [
        r'\b(b\.?tech|btech|m\.?tech|mtech|b\.?e\.?|be|m\.?e\.?|me|bsc|msc|bca|mca|ba|ma|phd|doctorate)\b',
        r'\bbachelor\s+(?:of|in)',
        r'\bmaster\s+(?:of|in)',
        r'\bdegree\s+(?:in|of)',
    ]
    
    has_degree_pattern = any(re.search(pattern, lw, re.IGNORECASE) for pattern in degree_patterns)
    has_education_keyword = any(keyword in lw for keyword in education_keywords)
    
    if has_degree_pattern or has_education_keyword:
        # STRICT: If it's clearly a degree format (e.g., "Btech, Civil Engineering"), reject it
        # Even if it contains role keywords like "engineering" - this is a degree, not a job title
        if has_degree_pattern:
            # Check if it's a degree followed by specialization (e.g., "Btech, Civil Engineering")
            if re.search(r'\b(b\.?tech|btech|m\.?tech|mtech|b\.?e\.?|be|m\.?e\.?|me|bsc|msc|bca|mca|ba|ma|phd|doctorate|bachelor|master)\s*[,\(]?\s*[a-z\s]+(?:engineering|science|arts|commerce|technology|computer|information)', lw, re.IGNORECASE):
                return True  # Definitely a degree format - reject
        
        # But allow if it's clearly a job title (e.g., "Education Manager" - has manager/director/etc)
        # AND doesn't have degree patterns
        if not has_degree_pattern and any(role in lw for role in ['manager', 'director', 'developer', 'consultant', 'analyst', 'architect', 'lead']):
            # Allow job titles like "Education Manager", "Engineering Director" (but not "Btech, Civil Engineering")
            return False
        
        # Also reject if it contains "of technology" or "institute of" (institution names)
        if re.search(r'\b(institute|institution|university|college)\s+of\s+', lw, re.IGNORECASE):
            return True
        
        # If it has education keywords but no clear job title role, reject
        if not any(role in lw for role in ['manager', 'director', 'developer', 'consultant', 'analyst', 'architect', 'lead']):
            return True
    
    # explicit invalid markers
    for token in NON_TITLE_INDICATORS:
        if token in lw:
            return True
    # description verbs suggest it is a sentence, not a title
    for v in DESCRIPTION_VERBS:
        # require whole-word match
        if re.search(r'\b' + re.escape(v) + r'\b', lw):
            return True
    # If the line contains many words (>10) it's unlikely to be a title
    # BUT: If it's in format "Title - Company (Date)" and contains "till date" or "present", 
    # extract just the title part for word count check
    word_count = len(lw.split())
    if word_count > 10:
        # Check if it's a "Title -/–/— Company (Date)" format with current date indicator
        if ('till date' in lw or 'present' in lw or 'current' in lw) and has_dash_separator(line):
            # Extract title part (before dash)
            title_part, _ = split_title_company(line)
            if title_part:
                title_word_count = len(title_part.lower().split())
                # If title part is reasonable (<=6 words), allow it
                if title_word_count <= 6:
                    return False  # Don't reject - it's a valid title format
        return True  # Reject if too many words and not in special format
    
    # Reject lines that are just "Leadership" or other standalone soft skills
    if lw in ['leadership', 'communication', 'teamwork', 'collaboration']:
        return True
    
    # Reject lines that look like sentence fragments (start with lowercase, no title keywords)
    if re.match(r'^[a-z]', line) and not any(title in lw for title in ['engineer', 'developer', 'consultant', 'manager', 'director', 'analyst', 'architect', 'lead', 'senior', 'sr']):
        return True
    
    return False


def best_match_from_known(line: str) -> Optional[str]:
    """
    Try exact-ish and fuzzy matches against COMMON_TITLES
    Optimized to prioritize longer/more specific matches first
    Preserves dashes from original text when present
    """
    # Preserve original line for dash checking
    original_line = line
    lw = re.sub(r'[^a-z0-9 ]', ' ', line.lower())
    lw = re.sub(r'\s+', ' ', lw).strip()
    if not lw:
        return None
    
    # Limit processing for very long lines (unlikely to be titles)
    if len(lw) > 200:
        return None

    # Check if original line contains dashes (for preservation)
    has_dash = '-' in original_line or '–' in original_line or '—' in original_line

    # Sort titles by length (longest first) to prioritize more specific matches
    # This ensures "Senior Consultant" matches before "Consultant"
    sorted_titles = sorted(COMMON_TITLES, key=len, reverse=True)
    
    # First pass: exact substring matches (prioritize longer matches)
    best_exact = None
    best_exact_len = 0
    best_exact_has_dash = False
    for known in sorted_titles:
        # Check if known title is contained in line (whole word match preferred)
        # Use word boundaries to avoid partial word matches
        pattern = r'\b' + re.escape(known) + r'\b'
        if re.search(pattern, lw, re.IGNORECASE):
            if len(known) > best_exact_len:
                best_exact = known
                best_exact_len = len(known)
                # Check if the known title has a dash version
                best_exact_has_dash = '-' in known or '–' in known or '—' in known
        # Also check if line is contained in known (for abbreviations)
        elif lw in known and len(known) > best_exact_len:
            best_exact = known
            best_exact_len = len(known)
            best_exact_has_dash = '-' in known or '–' in known or '—' in known
    
    if best_exact:
        # If original line has dash and we matched a non-dash version, try to find dash version
        if has_dash and not best_exact_has_dash:
            # Look for dash version of the matched title
            dash_variants = [
                best_exact.replace(' ', ' - '),
                best_exact.replace(' ', '-'),
                best_exact.replace(' ', ' – '),
            ]
            for variant in dash_variants:
                if variant.lower() in COMMON_TITLES:
                    return title_case(variant)
        return title_case(best_exact)
    
    # Second pass: fuzzy match (only if no exact match found)
    best_fuzzy = (None, 0.0, False)
    for known in sorted_titles:
        score = fuzzy_ratio(lw, known)
        if score > best_fuzzy[1]:
            has_dash_in_known = '-' in known or '–' in known or '—' in known
            best_fuzzy = (known, score, has_dash_in_known)
            # Early exit if we found a very good match
            if score >= 0.95:
                break

    if best_fuzzy[0] and best_fuzzy[1] >= FUZZY_THRESHOLD:
        # If original line has dash and we matched a non-dash version, try to find dash version
        if has_dash and not best_fuzzy[2]:
            # Look for dash version of the matched title
            dash_variants = [
                best_fuzzy[0].replace(' ', ' - '),
                best_fuzzy[0].replace(' ', '-'),
                best_fuzzy[0].replace(' ', ' – '),
            ]
            for variant in dash_variants:
                if variant.lower() in COMMON_TITLES:
                    return title_case(variant)
        return title_case(best_fuzzy[0])
    return None


def regex_extract_from_line(line: str) -> Optional[str]:
    """
    Regex heuristics to extract a designation token from a candidate line.
    E.g. picks "Senior Consultant" from "April 2023 to till date\nSenior Consultant"
    or from "Senior Consultant at RAMA corporate and IT solutions"
    """
    lw = line.lower()
    
    # CRITICAL: Reject degree patterns first (e.g., "Btech, Civil Engineering")
    degree_patterns = [
        r'\b(b\.?tech|btech|m\.?tech|mtech|b\.?e\.?|be|m\.?e\.?|me|bsc|msc|bca|mca|ba|ma|phd|doctorate)\s*[,\(]?\s*[a-z\s]+(?:engineering|science|arts|commerce|technology|computer|information)',
        r'\bbachelor\s+(?:of|in)\s+',
        r'\bmaster\s+(?:of|in)\s+',
        r'\bdegree\s+(?:in|of)\s+',
    ]
    
    # If line matches a degree pattern, reject it
    if any(re.search(pattern, lw, re.IGNORECASE) for pattern in degree_patterns):
        return None
    
    # look for common patterns: "Senior Consultant", "Consultant", "Technical Lead", etc.
    # Pattern: optional seniority + core role (removed "lead" from roles group to avoid duplication)
    roles_pattern = r'\b(?:(senior|sr|junior|jr|lead|principal|assistant|associate)\b[\s\.\-]*)?(' \
                    + r'engineer|developer|consultant|manager|architect|analyst|administrator|specialist|officer|director' \
                    + r')(?:\b|s\b)'
    m = re.search(roles_pattern, line, re.IGNORECASE)
    if m:
        groups = [g for g in m.groups() if g]
        cleaned = ' '.join(groups)
        return title_case(cleaned)
    # fallback: if line is short and contains 1-4 words and at least one core role keyword
    words = lw.split()
    if 1 <= len(words) <= 6 and any(k in lw for k in ['engineer', 'developer', 'consultant', 'manager', 'architect', 'lead', 'director', 'analyst']):
        # Additional check: reject if it looks like a degree (e.g., "Civil Engineering" after "Btech")
        if any(deg in lw for deg in ['btech', 'b.tech', 'mtech', 'm.tech', 'be', 'b.e', 'me', 'm.e', 'bachelor', 'master']):
            return None
        return title_case(lw)
    return None


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------
def extract_designation(resume_text: str) -> Optional[str]:
    """
    Main function. Given resume text, returns most-likely current designation or None.
    """
    if not resume_text or not resume_text.strip():
        return None

    text = normalize_text(resume_text)

    # Crop to experience-like section but be permissive
    exp_segment = crop_to_experience_section(text)

    # If cropped segment is very small, fallback to entire text
    if len(exp_segment) < 100:
        exp_segment = text

    # Collect prioritized candidate lines
    candidates = candidate_title_lines(exp_segment)
    
    # PRIORITIZE: First check for designations near "till date" or "present" dates
    # This ensures we get the CURRENT position, not just any position
    current_date_range_pattern = re.compile(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|[A-Za-z]+)\s+\d{4}\s+(?:to|till|-)\s+(?:till\s+date|date|present|current|till\s+now|ongoing)\b', re.IGNORECASE)
    current_date_keywords = re.compile(r'\b(till\s+date|to\s+date|present|current|till\s+now|ongoing)\b', re.IGNORECASE)
    
    exp_lines = exp_segment.split('\n')
    current_position_candidates = []
    
    # First pass: Find titles near "till date" or "present"
    for idx, line in enumerate(exp_lines):
        line_lower = line.lower()
        # Check if this line has a date with "till date" or "present"
        if current_date_range_pattern.search(line) or current_date_keywords.search(line_lower):
            # CRITICAL: Prioritize the line immediately BEFORE the date line (idx - 1)
            # This is where the job title typically appears (e.g., "Web Developer" before "Jun 2017 - Present")
            # Create prioritized list: [idx - 1] first, then other nearby lines
            priority_indices = []
            if idx > 0:
                priority_indices.append(idx - 1)  # Line immediately before date (highest priority)
            # Add other nearby lines (before and after)
            for check_idx in range(max(0, idx - 2), min(len(exp_lines), idx + 3)):
                if check_idx != idx and check_idx != idx - 1:  # Skip date line and already-added idx-1
                    priority_indices.append(check_idx)
            
            # Check lines in prioritized order
            for check_idx in priority_indices:
                check_line = exp_lines[check_idx].strip()
                if not check_line or check_line == line.strip():
                    continue
                if is_invalid_title_line(check_line):
                    continue
                
                # CRITICAL: Reject if this line appears to be in a certifications section
                check_line_lower = check_line.lower()
                check_line_pos = text.lower().find(check_line_lower)
                if check_line_pos > 0:
                    # Check text before this line for certifications header
                    text_before = text[:check_line_pos].lower()
                    # Look for certifications header (within last 500 chars to avoid false positives)
                    recent_text = text_before[-500:] if len(text_before) > 500 else text_before
                    cert_header_pattern = re.search(r'\b(certifications?|certificate|certified|cert\.?)\s*:?\s*$', recent_text, re.IGNORECASE | re.MULTILINE)
                    if cert_header_pattern:
                        # This line is likely in certifications section - skip it
                        continue
                
                # CRITICAL: For current positions, prioritize exact matches over fuzzy matches
                # First try exact match (case-insensitive)
                check_line_lower = check_line.lower().strip()
                exact_match = None
                # Use COMMON_TITLES from module scope (already defined at top of file)
                for known_title in COMMON_TITLES:
                    if known_title.lower() == check_line_lower:
                        exact_match = known_title
                        break
                
                if exact_match:
                    # Found exact match - return immediately (highest priority for current positions)
                    return title_case(exact_match)
                
                # If no exact match, try best_match_from_known (includes fuzzy matching)
                match = best_match_from_known(check_line) or regex_extract_from_line(check_line)
                if match:
                    # This is a current position - return it immediately
                    return match

    # Try each candidate in prioritized order
    for score, line in candidates:
        # Quick reject
        if is_invalid_title_line(line):
            continue
        
        # CRITICAL: Reject if this line appears to be in a certifications section
        # Check if line appears after "certifications" header in the original text
        line_lower = line.lower()
        line_pos = text.lower().find(line_lower)
        if line_pos > 0:
            # Check text before this line for certifications header
            text_before = text[:line_pos].lower()
            # Look for certifications header (within last 500 chars to avoid false positives)
            recent_text = text_before[-500:] if len(text_before) > 500 else text_before
            cert_header_pattern = re.search(r'\b(certifications?|certificate|certified|cert\.?)\s*:?\s*$', recent_text, re.IGNORECASE | re.MULTILINE)
            if cert_header_pattern:
                # This line is likely in certifications section - skip it
                continue

        # Handle "Title -/–/— Company (Date)" format - extract title part first
        # Check for all dash types: hyphen (-), en-dash (–), em-dash (—)
        if has_dash_separator(line) and ('till date' in line.lower() or 'present' in line.lower() or 'current' in line.lower() or re.search(r'\d{4}', line)):
            # Extract title part (before dash)
            title_part, _ = split_title_company(line)
            if title_part and not is_invalid_title_line(title_part):
                # First try exact match (prioritize shorter/exact matches)
                title_lower = title_part.lower()
                # Check if title_part exactly matches a known title (case-insensitive)
                # COMMON_TITLES is already defined at module level, no need to import
                for known_title in COMMON_TITLES:
                    if known_title.lower() == title_lower:
                        return title_case(known_title)
                # Then try fuzzy/substring match
                match = best_match_from_known(title_part) or regex_extract_from_line(title_part)
                if match:
                    # If match is longer than title_part, prefer title_part if it's a valid title
                    if len(match.split()) > len(title_part.split()) and any(kw in title_lower for kw in ['engineer', 'developer', 'consultant', 'manager', 'director', 'analyst', 'architect']):
                        # Use regex to extract from title_part directly
                        regex_match = regex_extract_from_line(title_part)
                        if regex_match:
                            return regex_match
                    return match
        
        # If a line contains 'at' or 'with' it's often "<Title> at <Company>"
        # extract left side
        if re.search(r'\b(at|with|@)\b', line, re.IGNORECASE):
            left = re.split(r'\b(?:at|with|@)\b', line, flags=re.IGNORECASE)[0].strip()
            # if left is short, attempt to match left
            if left and not is_invalid_title_line(left):
                match = best_match_from_known(left) or regex_extract_from_line(left)
                if match:
                    return match

        # 1) best match from known titles (fast)
        match = best_match_from_known(line)
        if match:
            return match

        # 2) regex heuristics
        match = regex_extract_from_line(line)
        if match:
            return match

    # Final fallback: search entire text for patterns like "\n<Title>\n" near company/date mentions
    # PRIORITIZE: Look for "till date" or "present" dates first (current positions)
    current_date_pattern = re.compile(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|[A-Za-z]+)\s+\d{4}\s+(?:to|till|-)\s+(?:till\s+date|date|present|current|till\s+now|ongoing)\b', re.IGNORECASE)
    lines = text.split('\n')
    
    # First pass: Look for current positions (till date/present)
    for idx, ln in enumerate(lines):
        if current_date_pattern.search(ln):
            # This is a current position - check nearby lines for title
            neighbors = lines[idx+1:idx+4] + lines[max(0, idx-3):idx]
            for neighbor in neighbors:
                if not neighbor or neighbor.strip() == ln.strip():
                    continue  # Skip empty lines and current line
                if is_invalid_title_line(neighbor):
                    continue
                m = best_match_from_known(neighbor) or regex_extract_from_line(neighbor)
                if m:
                    return m  # Return immediately for current position
    
    # Second pass: Look for any date patterns (fallback for resumes without "till date")
    date_like = re.compile(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|[A-Za-z]+)\s+\d{4}\b', re.IGNORECASE)
    for idx, ln in enumerate(lines):
        if date_like.search(ln):
            # check nearby lines (exclude current line explicitly)
            neighbors = lines[idx+1:idx+4] + lines[max(0, idx-3):idx]
            for neighbor in neighbors:
                if not neighbor or neighbor.strip() == ln.strip():
                    continue  # Skip empty lines and current line
                if is_invalid_title_line(neighbor):
                    continue
                m = best_match_from_known(neighbor) or regex_extract_from_line(neighbor)
                if m:
                    return m

    return None


# -------------------------------------------------------------------------
# Example usage (for quick local testing)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    sample = """
    RAMA corporate and IT solutions, Hyderabad
    April 2023 to till date
    Senior Consultant

    Project title : Sales order application including Approval
    Client : Burlington English
    """
    print("Detected designation:", extract_designation(sample))
