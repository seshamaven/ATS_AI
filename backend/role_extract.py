"""
Lightweight utilities to work with high-level roles and subroles.

This module is intentionally **category-agnostic** â€“ it only knows about
`role` and `subrole` (no category column).
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Sequence, Tuple, Dict

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Static role / subrole definitions
# -----------------------------------------------------------------------------

# NOTE: Legacy ROLE_SUBROLE_PAIRS removed; roles now come from ROLE_HIERARCHY
# and subroles are derived from skills, not a static mapping.
ROLE_SUBROLE_PAIRS: List[Tuple[str, str]] = []


# -----------------------------------------------------------------------------
# Role Hierarchy (Highest to Lowest Priority)
# -----------------------------------------------------------------------------

ROLE_HIERARCHY =[

# Level 0: Highest (Executive/Director)
    [
        "business development executive", "executive consultant", "program manager/director"
    ],

# Level 2: Senior Management
    [
        "Program Manager", "ai program manager", "ev program manager", "it program manager",
        "logistics program manager", "material program manager", "materials program manager",
        "program manager", "program manager,", "project/program manager", "sap program manager",
        "sap service management program manager", "sr cybersecurity program manager",
        "sr program manager", "sr. it program manager", "sr. program manager",
        "sr. technical program manager", "strategic program manager", "technical program manager"
    ],

# Level 3: Management
    [
        "sr project manager", "sr. project manager"
    ],

# Level 4: Project Management
    [
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
        "sw driver development project manager", "technical project manager"
    ],

# Level 5: Lead Roles
    [
        "Lead", "Technical Lead", "lead", "technical lead"
    ],

# Level 6: Analyst
    [
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
        "web system analyst", "windchill product analyst", "workforce planning analyst"
    ],

# Level 7: Senior Engineer
    [
        "sr .net developer", "sr .net web developer", "sr android developer", "sr c++ programmer",
        "sr data engineer", "sr developer", "sr java developer", "sr python full stack developer",
        "sr software engineer", "sr. .net developer", "sr. c# programmer", "sr. c++ developer",
        "sr. c++ programmer", "sr. developer", "sr. full stack engineer/architect",
        "sr. full stack engineer/architect1", "sr. fullstack developer", "sr. fullstack engineer",
        "sr. java developer", "sr. java programmer", "sr. net developer", "sr. programmer",
        "sr. python developer", "sr. software engineer" "sr .net back end developer",
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
    ],

# Level 8: Engineer (Lowest)
    [
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
    ],


]
def get_role_priority(role_or_subrole: str) -> int:
    """
    Get priority level for a role/subrole (lower number = higher priority).
    Returns 999 if not found in hierarchy.
    """
    text_lower = role_or_subrole.lower()
    for priority, role_list in enumerate(ROLE_HIERARCHY):
        for role in role_list:
            if role.lower() in text_lower or text_lower in role.lower():
                return priority
    return 999  # Not in hierarchy - lowest priority


# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------

def list_roles() -> List[str]:
    """Return the distinct list of high-level roles."""
    return sorted({role for role, _ in ROLE_SUBROLE_PAIRS})


def list_subroles(role: Optional[str] = None) -> List[str]:
    """
    Return all subroles, or subroles for a given role if provided.
    """
    if role is None:
        return sorted({sub for _, sub in ROLE_SUBROLE_PAIRS})
    return sorted({sub for r, sub in ROLE_SUBROLE_PAIRS if r.lower() == role.lower()})


def _normalize(text: str) -> str:
    """Normalize text for matching: lowercase, remove special chars, normalize whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[/,&]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_role_subrole(
    title_or_text: str,
    candidates: Sequence[Tuple[str, str]] = ROLE_SUBROLE_PAIRS,
) -> Optional[Tuple[str, str]]:
    """
    Detect role and subrole from text using exact values from ROLE_SUBROLE_PAIRS.
    When multiple matches are found, returns the highest priority role based on hierarchy.
    
    Matching strategy:
    1. Finds ALL matching roles/subroles
    2. Selects the one with highest priority (lowest hierarchy number)
    3. Returns exact values from ROLE_SUBROLE_PAIRS
    
    Returns exact values from ROLE_SUBROLE_PAIRS - no modifications.
    Category is deliberately ignored (only role and subrole are used).
    
    Args:
        title_or_text: Job title, designation, or text to search
        candidates: List of (role, subrole) tuples to match against
        
    Returns:
        Tuple of (role, subrole) if match found, None otherwise
    """
    if not title_or_text:
        return None

    text_norm = _normalize(title_or_text)
    matches = []  # List of (priority, role, subrole) tuples

    # Strategy 1: Check hierarchy first for high-priority roles (Director, VP, etc.)
    # This ensures "Director - Technology" matches before generic roles
    for priority, hierarchy_roles in enumerate(ROLE_HIERARCHY):
        for hierarchy_role in hierarchy_roles:
            hierarchy_norm = _normalize(hierarchy_role)
            # Check if hierarchy role appears in text
            if hierarchy_norm and hierarchy_norm in text_norm:
                # Find matching role/subrole pair that contains this hierarchy role
                for role, subrole in candidates:
                    role_norm = _normalize(role)
                    subrole_norm = _normalize(subrole)
                    if hierarchy_norm in role_norm or hierarchy_norm in subrole_norm:
                        matches.append((priority, role, subrole))
                        logger.debug(f"Matched hierarchy role '{hierarchy_role}' -> '{role}' / '{subrole}' (priority: {priority})")
                        break
                # If found a high-priority match, break early
                if matches and matches[-1][0] == 0:  # Priority 0 = highest
                    break
        if matches and matches[-1][0] == 0:  # Found highest priority, stop searching
            break

    # Strategy 2: Look for exact subrole phrase matches (more specific)
    for role, subrole in candidates:
        subrole_norm = _normalize(subrole)
        if subrole_norm and subrole_norm in text_norm:
            priority = get_role_priority(subrole)
            # Only add if not already found in hierarchy search
            if not any(r == role and s == subrole for _, r, s in matches):
                matches.append((priority, role, subrole))
                logger.debug(f"Matched subrole '{subrole}' (role: '{role}', priority: {priority})")

    # Strategy 3: Look for role phrase matches (less specific)
    for role, subrole in candidates:
        role_norm = _normalize(role)
        if role_norm and role_norm in text_norm:
            priority = get_role_priority(role)
            # Only add if not already found
            if not any(r == role and s == subrole for _, r, s in matches):
                matches.append((priority, role, subrole))
                logger.debug(f"Matched role '{role}' (subrole: '{subrole}', priority: {priority})")

    if not matches:
        logger.debug("No role/subrole match for text: %r", title_or_text[:100])
        return None

    # Sort by priority (lower number = higher priority) and return the highest
    matches.sort(key=lambda x: x[0])
    best_priority, best_role, best_subrole = matches[0]
    
    if len(matches) > 1:
        logger.debug(f"Found {len(matches)} role matches, selected highest priority: '{best_role}' / '{best_subrole}' (priority: {best_priority})")
    
    return best_role, best_subrole


def detect_all_roles(title_or_text: str) -> List[Tuple[int, str, str]]:
    """
    Find ALL roles in text using ROLE_HIERARCHY and return with priorities.
    Returns list of (priority, role, subrole) tuples sorted by priority.
    Subrole will be determined later from skills, so we return None for subrole here.
    """
    if not title_or_text:
        return []
    
    text_norm = _normalize(title_or_text)
    matches = []  # List of (priority, role, subrole) tuples
    
    # Map hierarchy roles to role_type names
    # This maps job titles from hierarchy to the actual role_type we want to store
    hierarchy_to_role_map = {
        # Level 0: Executive/Director -> Management & Product
        "technical director": "Management & Product",
        "director - technology": "Management & Product",
        "director technology": "Management & Product",
        "director": "Management & Product",
        "vice president": "Management & Product",
        "vp": "Management & Product",
        "chief technology officer": "Management & Product",
        "cto": "Management & Product",
        "chief executive officer": "Management & Product",
        "ceo": "Management & Product",
        "business development executive": "Management & Product",
        "executive consultant": "Management & Product",
        "program manager/director": "Management & Product",
        # Level 2: Senior Management -> Management & Product or preserve exact Consultant role names
        "delivery manager": "Management & Product",
        "senior delivery manager": "Management & Product",
        "sr delivery manager": "Management & Product",
        "program manager": "Management & Product",
        # Consultant roles - preserve exact names
        "senior consultant": "Senior Consultant",
        "sr consultant": "Senior Consultant",
        "sr. consultant": "Senior Consultant",
        "consultant": "Consultant",
        "technical consultant": "Technical Consultant",
        # Level 3: Management -> Management & Product
        "sr manager": "Management & Product",
        "senior manager": "Management & Product",
        "account manager": "Management & Product",
        "sr project manager": "Management & Product",
        "senior project manager": "Management & Product",
        # Level 4: Project Management -> Management & Product
        "project manager": "Management & Product",
        # Level 5: Lead Roles -> Software Engineer (or Management & Product for some)
        "project lead": "Software Engineer",
        "technical lead": "Software Engineer",
        "tech lead": "Software Engineer",
        "module lead": "Software Engineer",
        "team lead": "Software Engineer",
        "lead": "Software Engineer",
        # Level 6: Analyst -> Business Analyst or Data Analyst
        "analyst": "Business Analyst",
        "business analyst": "Business Analyst",
        "data analyst": "Data Analyst",
        "system analyst": "Business Analyst",
        # Level 7: Senior Engineer -> Software Engineer
        "sr software engineer": "Software Engineer",
        "senior software engineer": "Software Engineer",
        "sr engineer": "Software Engineer",
        "senior engineer": "Software Engineer",
        # Level 8: Engineer -> Software Engineer
        "software engineer": "Software Engineer",
        "engineer": "Software Engineer",
        "developer": "Software Engineer",
    }
    
    # Check hierarchy roles directly
    for priority, hierarchy_roles in enumerate(ROLE_HIERARCHY):
        for hierarchy_role in hierarchy_roles:
            hierarchy_norm = _normalize(hierarchy_role)
            if hierarchy_norm and hierarchy_norm in text_norm:
                # For consultant roles, preserve the exact role name from hierarchy
                # For other roles, use the mapping
                if "consultant" in hierarchy_norm:
                    # Preserve exact role name with proper capitalization (Title Case)
                    # Convert "senior consultant" -> "Senior Consultant"
                    role_type = ' '.join(word.capitalize() for word in hierarchy_role.split())
                else:
                    # Map hierarchy role to role_type
                    role_type = hierarchy_to_role_map.get(hierarchy_norm, "Software Engineer")
                
                # Subrole will be determined from skills later, so use None here
                if not any(r == role_type for _, r, _ in matches):
                    matches.append((priority, role_type, None))
    
    # Sort by priority (lower = higher priority)
    matches.sort(key=lambda x: x[0])
    return matches


def match_subrole_from_skills(role: str, primary_skills: str, secondary_skills: str = "") -> Optional[str]:
    """
    Match subrole based on skills. Always returns one of: "Backend Developer", 
    "Full Stack Developer", or "Frontend Developer".
    
    Args:
        role: The selected role
        primary_skills: Comma-separated primary/technical skills
        secondary_skills: Comma-separated secondary skills
        
    Returns:
        One of: "Backend Developer", "Full Stack Developer", or "Frontend Developer"
    """
    if not primary_skills:
        return "Backend Developer"  # Default
    
    # Combine all skills for matching
    all_skills_text = f"{primary_skills}, {secondary_skills}".lower()
    all_skills_list = [s.strip().lower() for s in all_skills_text.split(',') if s.strip()]
    
    # Define skill keywords for each subrole type
    frontend_keywords = [
        'react', 'angular', 'vue', 'javascript', 'typescript', 'html', 'css', 
        'sass', 'scss', 'less', 'jsx', 'tsx', 'next.js', 'nuxt', 'gatsby',
        'webpack', 'vite', 'frontend', 'front-end', 'ui', 'ux', 'jquery',
        'bootstrap', 'tailwind', 'material-ui', 'ant design'
    ]
    
    backend_keywords = [
        'node.js', 'nodejs', 'express', 'django', 'flask', 'fastapi', 'spring',
        'spring boot', 'hibernate', 'api', 'rest', 'graphql', 'microservices',
        'server', 'backend', 'back-end', 'serverless', 'lambda', 'asp.net',
        '.net', 'c#', 'java', 'python', 'php', 'ruby', 'rails', 'go', 'golang',
        'rust', 'scala', 'kotlin', 'database', 'sql', 'nosql', 'mongodb',
        'postgresql', 'mysql', 'redis', 'elasticsearch', 'kafka', 'rabbitmq'
    ]
    
    # Count matches for frontend and backend
    frontend_score = sum(1 for keyword in frontend_keywords if keyword in all_skills_text)
    backend_score = sum(1 for keyword in backend_keywords if keyword in all_skills_text)
    
    # Determine subrole based on scores
    if frontend_score > 0 and backend_score > 0:
        # Has both frontend and backend skills -> Full Stack
        return "Full Stack Developer"
    elif frontend_score > backend_score:
        # More frontend skills -> Frontend
        return "Frontend Developer"
    else:
        # More backend skills or equal -> Backend (default)
        return "Backend Developer"


def determine_subrole_type_from_profile_and_skills(
    profile_type: str,
    primary_skills: str,
    secondary_skills: str = "",
) -> str:
    """
    Determine subrole_type based on profile_type and skills.
    Always returns one of: "Backend Developer", "Full Stack Developer", or "Frontend Developer".
    This is used to populate subrole_type (not sub_profile_type).
    """
    if not profile_type or not primary_skills:
        return "Backend Developer"  # Default
    
    # Combine all skills for matching
    all_skills_text = f"{primary_skills}, {secondary_skills}".lower()
    
    # Define skill keywords
    frontend_keywords = [
        'react', 'angular', 'vue', 'javascript', 'typescript', 'html', 'css', 
        'sass', 'scss', 'less', 'jsx', 'tsx', 'next.js', 'nuxt', 'gatsby',
        'webpack', 'vite', 'frontend', 'front-end', 'ui', 'ux', 'jquery',
        'bootstrap', 'tailwind', 'material-ui', 'ant design'
    ]
    
    backend_keywords = [
        'node.js', 'nodejs', 'express', 'django', 'flask', 'fastapi', 'spring',
        'spring boot', 'hibernate', 'api', 'rest', 'graphql', 'microservices',
        'server', 'backend', 'back-end', 'serverless', 'lambda', 'asp.net',
        '.net', 'c#', 'java', 'python', 'php', 'ruby', 'rails', 'go', 'golang',
        'rust', 'scala', 'kotlin', 'database', 'sql', 'nosql', 'mongodb',
        'postgresql', 'mysql', 'redis', 'elasticsearch', 'kafka', 'rabbitmq'
    ]
    
    # Count matches for frontend and backend skills
    frontend_score = sum(1 for keyword in frontend_keywords if keyword in all_skills_text)
    backend_score = sum(1 for keyword in backend_keywords if keyword in all_skills_text)
    
    # Normalize profile_type for comparison
    profile_lower = profile_type.lower()
    
    # Categorize profile types
    backend_profiles = ['java', '.net', 'python', 'c#', 'php', 'ruby', 'go', 'golang', 'rust', 'scala', 'kotlin']
    frontend_profiles = ['javascript', 'typescript', 'ui/ux', 'ui', 'ux', 'html', 'css']
    # Non-technical profiles that shouldn't get developer subroles
    non_technical_profiles = ['business development', 'sales', 'marketing', 'hr', 'recruitment', 'accounting', 'finance', 'legal', 'support', 'operations']
    
    # Check if profile_type is backend-focused
    is_backend_profile = any(bp in profile_lower for bp in backend_profiles)
    # Check if profile_type is frontend-focused
    is_frontend_profile = any(fp in profile_lower for fp in frontend_profiles)
    # Check if profile_type is non-technical
    is_non_technical = any(ntp in profile_lower for ntp in non_technical_profiles)
    
    # Determine subrole_type based on profile_type and skills
    if is_backend_profile:
        # Backend profile (Java, .Net, Python, etc.)
        if frontend_score > 0:
            # Has frontend skills -> Full Stack
            return "Full Stack Developer"
        else:
            # No frontend skills -> Backend
            return "Backend Developer"
    elif is_frontend_profile:
        # Frontend profile (JavaScript, UI/UX, etc.)
        if backend_score > 0:
            # Has backend skills -> Full Stack
            return "Full Stack Developer"
        else:
            # No backend skills -> Frontend
            return "Frontend Developer"
    elif is_non_technical:
        # Non-technical profiles - default to Backend Developer (or could be None, but we need one of the three)
        # If they have strong technical skills, determine from skills
        if frontend_score > 0 and backend_score > 0:
            return "Full Stack Developer"
        elif frontend_score > backend_score and frontend_score >= 2:  # Need at least 2 frontend skills
            return "Frontend Developer"
        else:
            return "Backend Developer"  # Default for non-technical
    else:
        # Other technical profiles (Data Science, DevOps, etc.) - determine from skills
        if frontend_score > 0 and backend_score > 0:
            return "Full Stack Developer"
        elif frontend_score > backend_score:
            return "Frontend Developer"
        else:
            return "Backend Developer"


def is_non_it_profile(profile_type: str) -> bool:
    """
    Check if a profile_type is non-IT related (Business Development, Sales, Marketing, etc.).
    
    Args:
        profile_type: The profile type to check
        
    Returns:
        True if non-IT profile, False otherwise
    """
    if not profile_type:
        return False
    
    profile_lower = profile_type.lower()
    
    # Non-IT profile types
    non_it_profiles = [
        'business development', 'sales', 'marketing', 'hr', 'human resources',
        'recruitment', 'recruiter', 'talent acquisition', 'accounting', 'finance',
        'financial', 'legal', 'support', 'operations', 'customer service',
        'customer support', 'admin', 'administration', 'executive assistant',
        'business analyst', 'functional analyst', 'process analyst', 'product analyst',
        'teacher', 'lecturer', 'professor', 'education', 'doctor', 'physician',
        'nurse', 'medical', 'healthcare', 'banking', 'insurance'
    ]
    
    return any(ntp in profile_lower for ntp in non_it_profiles)


def infer_role_from_skills(primary_skills: str, secondary_skills: str = "", profile_type: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """
    Infer role and subrole purely from skills (fallback when no role found in resume).
    Subrole is always one of: "Backend Developer", "Full Stack Developer", or "Frontend Developer".
    
    Args:
        primary_skills: Comma-separated primary/technical skills
        secondary_skills: Comma-separated secondary skills
        profile_type: Optional profile type to check if non-IT
        
    Returns:
        Tuple of (role, subrole) if inferred, None otherwise (returns None for non-IT profiles)
    """
    # If profile_type is non-IT, don't infer IT roles
    if profile_type and is_non_it_profile(profile_type):
        return None
    
    if not primary_skills:
        return None
    
    all_skills_text = f"{primary_skills}, {secondary_skills}".lower()
    all_skills_list = [s.strip().lower() for s in all_skills_text.split(',') if s.strip()]
    
    # Determine subrole first (always one of the three)
    frontend_keywords = [
        'react', 'angular', 'vue', 'javascript', 'typescript', 'html', 'css', 
        'sass', 'scss', 'less', 'jsx', 'tsx', 'next.js', 'nuxt', 'gatsby',
        'webpack', 'vite', 'frontend', 'front-end', 'ui', 'ux', 'jquery',
        'bootstrap', 'tailwind', 'material-ui', 'ant design'
    ]
    
    backend_keywords = [
        'node.js', 'nodejs', 'express', 'django', 'flask', 'fastapi', 'spring',
        'spring boot', 'hibernate', 'api', 'rest', 'graphql', 'microservices',
        'server', 'backend', 'back-end', 'serverless', 'lambda', 'asp.net',
        '.net', 'c#', 'java', 'python', 'php', 'ruby', 'rails', 'go', 'golang',
        'rust', 'scala', 'kotlin', 'database', 'sql', 'nosql', 'mongodb',
        'postgresql', 'mysql', 'redis', 'elasticsearch', 'kafka', 'rabbitmq'
    ]
    
    frontend_score = sum(1 for keyword in frontend_keywords if keyword in all_skills_text)
    backend_score = sum(1 for keyword in backend_keywords if keyword in all_skills_text)
    
    # Determine subrole
    if frontend_score > 0 and backend_score > 0:
        subrole = "Full Stack Developer"
    elif frontend_score > backend_score:
        subrole = "Frontend Developer"
    else:
        subrole = "Backend Developer"
    
    # Skill-based role inference patterns (role only, subrole is already determined)
    skill_patterns = {
        # Data Science / ML
        'Data Scientist': ['python', 'machine learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn', 'keras', 'neural network', 'deep learning', 'ai', 'artificial intelligence', 'data science'],
        'Data Engineer': ['etl', 'data pipeline', 'airflow', 'spark', 'hadoop', 'kafka', 'snowflake', 'data engineering'],
        'Data Analyst': ['data analysis', 'pandas', 'numpy', 'sql', 'excel', 'tableau', 'power bi', 'bi', 'business intelligence'],
        
        # Software Engineering
        'Software Engineer': ['.net', 'c#', 'csharp', 'asp.net', 'dotnet', 'java', 'spring', 'hibernate', 'python', 'django', 'flask', 'javascript', 'typescript', 'react', 'angular', 'vue', 'node.js', 'express', 'software', 'developer', 'programming'],
        
        # DevOps / Cloud
        'DevOps Engineer': ['devops', 'ci/cd', 'jenkins', 'gitlab', 'docker', 'kubernetes', 'terraform', 'ansible', 'chef', 'puppet'],
        'Cloud Engineer': ['aws', 'azure', 'gcp', 'cloud', 'ec2', 's3', 'lambda', 'cloud computing'],
        
        # QA / Testing
        'QA Engineer': ['selenium', 'cypress', 'test automation', 'qa', 'testing', 'junit', 'pytest', 'quality assurance'],
        
        # Database
        'Database Administrator': ['sql', 'mysql', 'postgresql', 'oracle', 'database', 'dba', 'database administration'],
        
        # Mobile
        'Mobile Developer': ['android', 'ios', 'flutter', 'react native', 'mobile', 'swift', 'kotlin', 'xamarin'],
        
        # SAP / ERP
        'SAP Consultant': ['sap', 'abap', 'hana', 'sap fico', 'sap mm', 'sap sd'],
    }
    
    # Score each role pattern
    role_scores = []
    for role, keywords in skill_patterns.items():
        score = sum(10 if keyword in all_skills_text else 0 for keyword in keywords)
        if score > 0:
            role_scores.append((score, role))
    
    if not role_scores:
        # Don't default to Software Engineer - return None if no IT skills found
        # This prevents assigning IT roles to non-IT profiles
        return None
    
    # Return highest scoring role with the determined subrole
    role_scores.sort(key=lambda x: x[0], reverse=True)
    return role_scores[0][1], subrole


def detect_role_only(title_or_text: str) -> Optional[str]:
    """Convenience helper to get just the role from text."""
    result = detect_role_subrole(title_or_text)
    return result[0] if result else None


def detect_subrole_only(title_or_text: str) -> Optional[str]:
    """Convenience helper to get just the subrole from text."""
    result = detect_role_subrole(title_or_text)
    return result[1] if result else None


__all__ = [
    "ROLE_SUBROLE_PAIRS",
    "ROLE_HIERARCHY",
    "list_roles",
    "list_subroles",
    "detect_role_subrole",
    "detect_all_roles",
    "match_subrole_from_skills",
    "determine_subrole_type_from_profile_and_skills",
    "is_non_it_profile",
    "infer_role_from_skills",
    "detect_role_only",
    "detect_subrole_only",
]


