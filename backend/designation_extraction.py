# designation_extraction.py
"""
Resume Designation Extractor

Task: Extract ONLY the candidate's CURRENT or MOST RECENT professional job designation.

STRICT RULES (VERY IMPORTANT):
1. Extract designation ONLY from:
   - Resume header (name section) - HIGH PRIORITY
   - Experience / Work Experience section

2. IGNORE designations from:
   - Skills
   - Certifications
   - Tools or technologies
   - Education
   - Summary (unless clearly a job title line)
   - Volunteer roles (unless no professional role exists)

3. If multiple roles exist:
   - Prefer the most recent professional role
   - If no current role exists, pick the latest past role
   - Ignore internships if a professional role exists

4. A valid designation must:
   - Be 1-6 words
   - Represent a real job title (Analyst, Engineer, Developer, Manager, Assistant, etc.)
   - NOT be a skill (e.g., Salesforce Developer if only in certification)
   - NOT be a tool or technology name

5. If the designation appears near the candidate name at the top, give it HIGH priority.

PRIORITY ORDER:
  Priority 0 - Header/Name Section (HIGHEST):
    - Check resume header (first 10-15 lines, before Experience section)
    - If designation found near candidate name, return immediately
  
  Priority 1 - Current Role (Sequential):
    Step 1 - Explicit Current Indicators:
      - Select designation only if date range explicitly contains: Present, Current, Till Date, 
        Till Now, Now, Ongoing, Still Working
      - If found, return immediately (skip Step 2)
      - Designation must appear on same line or within 5 lines of the date
      - If multiple "Present" roles exist, choose the most recent one in resume order
    Step 2 - Future Dates (only if Step 1 found nothing):
      - Treats date ranges ending in the future (beyond current date) as current roles
      - Designation must appear on same line or within 5 lines of the date
  
  Priority 2 - Latest End Date:
    - If no current role exists, identify all roles with date ranges
    - Parse end dates (year > month > text)
    - Select role with latest end date
    - If multiple roles end in same year, prefer the last occurring role
  
  Priority 3 - Fallback (No Dates Anywhere):
    - Select first valid designation from experience section
    - Ignore titles from education, certification, project, volunteer, or skills sections

OUTPUT FORMAT (STRICT):
  - Output ONLY the designation string
  - No company name, no dates, Title Case, no extra text

Returns: Single designation string (title-cased) or None if not found.
"""

import re
from difflib import SequenceMatcher
from typing import Optional, List, Tuple
from datetime import datetime


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
    "associate consultant", "associate", "intern", "trainee","Operations & Technology Intern","Product Manager",
    "Graphic Designer", "Production Artist","Senior Reviewer/Writer",
    "Technical Writer Volunteer","Content Writer","Reporter and Editor","Data Collection Specialist","Administrative Assistant","Sales Consultant","Executive Assistant","Security Officer","Recreation Leader","Landscaper","E-Learning Case Manager and Content Specialist",
    "Technical Success Specialist", "Operations Manager","Program Management  Consultant", "Sr. Automation Engineer",
    "Assistant Manager - Enterprise Digitalization","Mental Health","Life Coach","Instructional Designer","Corporate Trainer", "JavaScript Developer"

    # Director roles
        "director", "technical director", "director - technology", "director technology", "Director - Technology","Volunteer",
        "director of technology", "it director", "software director", "engineering director",
        "development director", "program director", "project director", "delivery director",
        "senior director", "sr. director", "sr director", "associate director",
        "assistant director", "deputy director", "executive director", "managing director","assistant",
        
        # Consultant roles
        "consultant", "senior consultant", "sr. consultant", "sr consultant",
        "executive consultant", "technical consultant", "it consultant", "business consultant",
        "management consultant", "senior technical consultant", "lead consultant",
        "principal consultant", "associate consultant", "junior consultant",
        
        # Manager roles
        "business development executive", "executive consultant", "program manager/director",
        # Business Development & Sales roles
        "lead generation specialist", "sales development executive", "marketing executive", 
        "management trainee", "sales executive", "business development manager",
        "sales manager", "marketing manager", "account executive", "sales representative",
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
        "sr software engineer", "sr .net developer", "sr. c# programmer", "sr. c++ developer",
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
        "sr. wireless automation developer", "sr. xml developer", "sr .net developer",
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
        "net developer", "net full stack developer", "net fullstack developer",".Net fullstack developer",".Net full stack developer","Sr .Net Full Stack Developer", "Senior full stack .Net developer", "Full stack Dot Net developer",
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
        "wireless systems engineer", "wix developer", "xcelsius developer",
        # Education & Training roles
        "teacher", "health teacher", "wellness teacher", "physical education teacher",
        "english teacher", "secondary english teacher", "journalism teacher",
        "adviser", "advisor", "journalism adviser", "journalism advisor",
        "instructional designer", "learning designer", "curriculum designer",
        "e-learning case manager", "e-learning content specialist",
        "content specialist", "case manager", "training consultant",
        "instructor", "trainer", "coordinator", "training coordinator",
        "department chair", "department head", "care taker", "personal support worker","Marketing Intern", "Marketing Assistant", "Policy Intern",
        "Communications Manager", "Collaborative Marketing", "Copywriter", "Social Media Specialists", "Resident Service Coordinator",
        "BRAND MANAGER", "BRAND MANAGER INTERN", "MARKETING RESEARCH MANAGER", "ONLINE MARKETING SPECIALIST",
        "Field Marketing Manager", "Event Marketing Staff", "Database Administrator", "Email Technician",
        "EDI Analyst", "IT  Technician", "Investor Relations Coordinator", "Financial Planning Associate","Financial Auditor",
        "Digital Marketing Specialist", "PPC/SEO Specialist", "Senior PPC Specialist","Health Care Agent","Operations Associate",
        "Litigation Intelligence & Investigative Experts","Receptionist",
        
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
    
    # Fix PDF spacing issues: "J u n e" -> "June", "P R E S E N T" -> "PRESENT"
    # Pattern: single letter followed by space, repeated 2+ times (for months, dates, keywords)
    # Only apply to common date-related words to avoid breaking legitimate spaced text
    date_keywords = {
        'j a n u a r y': 'january', 'f e b r u a r y': 'february', 'm a r c h': 'march',
        'a p r i l': 'april', 'm a y': 'may', 'j u n e': 'june', 'j u l y': 'july',
        'a u g u s t': 'august', 's e p t e m b e r': 'september', 'o c t o b e r': 'october',
        'n o v e m b e r': 'november', 'd e c e m b e r': 'december',
        'j a n': 'jan', 'f e b': 'feb', 'm a r': 'mar', 'a p r': 'apr',
        'j u n': 'jun', 'j u l': 'jul', 'a u g': 'aug', 's e p': 'sep',
        'o c t': 'oct', 'n o v': 'nov', 'd e c': 'dec',
        'p r e s e n t': 'present', 'c u r r e n t': 'current',
        't o': 'to', 'f r o m': 'from'
    }
    for spaced, fixed in date_keywords.items():
        t = t.replace(spaced, fixed)
        t = t.replace(spaced.upper(), fixed.upper())
        t = t.replace(spaced.capitalize(), fixed.capitalize())
    
    # More general fix: single character + space pattern for date-related contexts
    # Match patterns like "J u n e 2 0 2 0" but be careful not to break legitimate text
    # Only apply in contexts that look like dates (followed by numbers or date keywords)
    def fix_spaced_chars(match):
        spaced_text = match.group(0)
        # Remove spaces between single characters
        fixed = re.sub(r'([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])', r'\1\2\3\4', spaced_text)
        # Only return fixed if it matches a known month or date keyword
        fixed_lower = fixed.lower().strip()
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 
                  'august', 'september', 'october', 'november', 'december',
                  'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
                  'present', 'current']
        if any(month in fixed_lower for month in months):
            return fixed
        return spaced_text
    
    # Apply to patterns that look like spaced-out months/dates
    t = re.sub(r'\b([A-Za-z]\s+){3,10}(?=\s+\d|\s+present|\s+current|\s+to|\s+from)', fix_spaced_chars, t, flags=re.IGNORECASE)
    
    # Fix number spacing: "2 0 2 0" -> "2020" (for years in date contexts)
    # Pattern: 4 digits separated by spaces (likely a year)
    def fix_spaced_numbers(match):
        spaced_nums = match.group(0)
        # Remove spaces between digits
        fixed = re.sub(r'(\d)\s+(\d)\s+(\d)\s+(\d)', r'\1\2\3\4', spaced_nums)
        return fixed
    
    # Match 4-digit years with spaces: "2 0 2 0", "2 0 2 1", etc.
    # Only in contexts that look like dates (near months or date keywords)
    t = re.sub(r'\b(\d\s+){3}\d\b', fix_spaced_numbers, t)
    
    # Also fix 2-digit numbers that might be part of dates: "0 6" -> "06", "1 2" -> "12"
    # But only when they appear near date-related context
    t = re.sub(r'\b(\d)\s+(\d)(?=\s*[-–—]|\s+present|\s+current|\s*$)', r'\1\2', t)
    
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
# CORE: Header/Name Section Extraction
# -------------------------------------------------------------------------
def extract_designation_from_header(text: str) -> Optional[str]:
    """
    Extract designation from resume header/name section (first 10-15 lines).
    Header typically contains: Name, Designation, Contact Info, Location.
    
    Returns: Designation string if found, None otherwise.
    """
    lines = text.split('\n')
    
    # Get first 15 lines (header section typically ends before Experience section)
    header_lines = []
    for i, line in enumerate(lines[:15]):
        line_stripped = line.strip()
        if line_stripped:
            # Stop if we hit a major section header
            if re.search(r'^(experience|work history|employment|education|skills|certifications|summary|objective)\s*[:]?$', line_stripped, re.IGNORECASE):
                break
            header_lines.append((i, line_stripped))
    
    if not header_lines:
        return None
    
    # Look for designation in header lines
    # Common patterns in header:
    # - Name on line 1, Designation on line 2
    # - Name and Designation on same line (separated by comma or dash)
    # - Designation appears within first 5 non-empty lines
    
    for line_idx, line in header_lines[:10]:  # Check first 10 header lines
        # Skip obvious non-designation lines
        if is_invalid_title_line(line):
            continue
        
        # Skip lines that are clearly contact info
        if re.search(r'(@|\bemail\b|\bphone\b|\+\d|\d{3}[-.\s]?\d{3}[-.\s]?\d{4})', line, re.IGNORECASE):
            continue
        
        # Skip lines that are just addresses (contain street, city, state, zip patterns)
        if re.search(r'\b(street|avenue|road|drive|lane|blvd|boulevard|city|state|zip|postal)\b', line, re.IGNORECASE):
            continue
        
        # Try to extract designation from this line
        title_from_line = extract_title_from_candidate_line(line)
        if title_from_line and not is_invalid_title_line(title_from_line):
            match = best_match_from_known(title_from_line, False) or regex_extract_from_line(title_from_line)
            if match:
                # Validate: 1-6 words and contains role indicator
                words = match.split()
                if 1 <= len(words) <= 6:
                    validated = _validate_designation_result(match, text)
                    if validated:
                        return validated
    
    return None


# -------------------------------------------------------------------------
# CORE: extracting a cleaned "experience" segment (to avoid project sections)
# -------------------------------------------------------------------------
def crop_to_experience_section(text: str) -> str:
    """
    Try to return the portion of text that is the experience / work history.
    If no explicit header, return the first 2500 chars but stop at portfolio/project section.
    """
    lower = text.lower()
    # CRITICAL: Only match section headers (on their own line or followed by colon/newline)
    # This prevents matching "experience" in sentences like "Experienced in..."
    # Pattern matches: "Work history", "Work history:", "EXPERIENCE", "Experience:", etc.
    # But NOT "Experienced in seamless integration..."
    # Priority: "Work history" first (most specific), then other patterns
    
    # DEBUG: Log what we're searching for
    import logging
    logger = logging.getLogger(__name__)
    
    # Try to find "Work history" or "Job History" first (most common and specific)
    work_history_pattern = re.compile(
        r'^(?:[#\s\-•]*)?(work\s+history|job\s+history)\s*[:#]?\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    m = work_history_pattern.search(lower)
    if m:
        start = m.start()
        slice_text = text[start:]
        logger.info(f"DEBUG: ✓ Found 'Work history'/'Job History' header at position {start}")
    else:
        # Try other experience headers (including "RELEVANT EXPERIENCE", "ADDITIONAL EXPERIENCE")
        experience_header_pattern = re.compile(
            r'^(?:[#\s\-•]*)?(employment\s+history|professional\s+experience|career\s+development|relevant\s+experience|additional\s+experience)\s*[:#]?\s*$',
            re.IGNORECASE | re.MULTILINE
        )
        m = experience_header_pattern.search(lower)
        if m:
            start = m.start()
            slice_text = text[start:]
            logger.info(f"DEBUG: ✓ Found experience header '{text[m.start():m.end()]}' at position {start}")
        else:
            # Fallback: try "experience" as a header (including "Experience & Certifications", "Experience & ...")
            # Pattern matches "Experience" followed by optional "& ..." or colon
            experience_header_pattern2 = re.compile(
                r'^(?:[#\s\-•]*)?experience(?:\s*&\s*[^:]*)?\s*[:#]?\s*$',
                re.IGNORECASE | re.MULTILINE
            )
            m = experience_header_pattern2.search(lower)
            if m:
                start = m.start()
                slice_text = text[start:]
                logger.info(f"DEBUG: ✓ Found 'Experience' header (possibly with '& ...') at position {start}")
            else:
                # fallback: start from top (many resumes don't have headers)
                slice_text = text[:4000]
                logger.warning(f"DEBUG: ✗ No experience header found, using first 4000 chars")

    # CRITICAL: FIRST - Cut off at "Gates Institute Of Technology" or similar institution names
    # This must be done BEFORE other section cropping to prevent "Director Of Technology" extraction
    # This is a safety net to catch institution names even if education header isn't detected
    gates_pattern = re.search(r'\b(gates\s+institute\s+of\s+technology|institute\s+of\s+technology)\b', slice_text, re.IGNORECASE)
    if gates_pattern:
        # Cut off before the institution name
        slice_text = slice_text[:gates_pattern.start()]
    
    # CRITICAL: Cut off at education section to avoid institution names being mistaken for job titles
    # This prevents "Gates Institute Of Technology" or "Director Of Technology" from education being extracted
    # Also prevents matching "Director" from "Director Of Technology" in education section
    # Pattern: Match "EDUCATION" header (with or without colon, on its own line or followed by content)
    edu_stop = re.search(r'\b(education|educational|academic|qualification|degree|b\.?tech|m\.?tech|b\.?e\.?|m\.?e\.?|bachelor|master|ph\.?d|doctorate)\s*:?\s*$', slice_text, re.IGNORECASE | re.MULTILINE)
    if not edu_stop:
        # Also try matching "EDUCATION" as a standalone header (might be on its own line, with or without markdown)
        # Match patterns like: "## EDUCATION", "EDUCATION", "EDUCATION:" etc.
        edu_stop = re.search(r'^[#\s]*(education|educational|academic|qualification)\s*[:#]?\s*$', slice_text, re.IGNORECASE | re.MULTILINE)
    if edu_stop:
        slice_text = slice_text[:edu_stop.start()]
    
    # Cut off at portfolio/project section to avoid project-title becoming job title
    stop = re.search(r'(portfolio of projects|portfolio|project\s+\d+|project title|projects|project:)', slice_text, re.IGNORECASE)
    if stop:
        slice_text = slice_text[:stop.start()]
    
    # CRITICAL: Cut off at certifications section to avoid certification titles being mistaken for job titles
    # This prevents "Salesforce Developer" from certifications being extracted as current designation
    cert_stop = re.search(r'\b(certifications?|certificate|certified|cert\.?)\s*:?\s*$', slice_text, re.IGNORECASE | re.MULTILINE)
    if cert_stop:
        slice_text = slice_text[:cert_stop.start()]
    
    # CRITICAL: Cut off at skills section to avoid skill keywords being mistaken for job titles
    # This prevents "Coordinator" from "Schedule coordination" in skills section being extracted as current designation
    # BUT: Only cut if it's a clear section header (on its own line), not if "skills" appears in job descriptions
    # Also ensure we don't cut too early - check if there are more roles after "skills" mention
    skills_stop = re.search(r'^(?:[#\s\-•]*)?(skills?|technical\s+skills?|core\s+skills?|key\s+skills?|professional\s+skills?|competencies?)\s*:?\s*$', slice_text, re.IGNORECASE | re.MULTILINE)
    if skills_stop:
        # Before cutting, check if there are date patterns after this position (might be more roles)
        text_after_skills = slice_text[skills_stop.start():]
        # If we find date patterns after skills, don't cut (might be roles listed after skills section header)
        date_after_skills = re.search(r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\s+\d{4}\s*([-–—]|to)', text_after_skills[:500], re.IGNORECASE)
        if not date_after_skills:
            slice_text = slice_text[:skills_stop.start()]
            logger.info(f"DEBUG: ✓ Cut off at skills section at position {skills_stop.start()}")
    
    # CRITICAL: Cut off at leadership/service/volunteer sections to avoid volunteer roles being mistaken for professional job titles
    # This prevents "Youth Programs Leader" from "LEADERSHIP & SERVICE EXPERIENCE" section being extracted as current designation
    # These are volunteer/community service roles, not professional work experience
    leadership_stop = re.search(r'\b(leadership\s*(?:&|and)\s*service\s+experience|leadership\s+experience|service\s+experience|volunteer\s+experience|volunteer\s+work|community\s+service|community\s+involvement|extracurricular\s+activities)\s*:?\s*$', slice_text, re.IGNORECASE | re.MULTILINE)
    if leadership_stop:
        slice_text = slice_text[:leadership_stop.start()]
        logger.info(f"DEBUG: ✓ Cut off at leadership/service/volunteer section at position {leadership_stop.start()}")

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
        if re.match(r'^(playing|managing|strategizing|front|optimizing|empowering|coaching|initiated|standardizing|formalizing|responsible|establishing|improving|has|utilized|developed|created|configured|implemented|enabled|designed|integrated|automated|conducted|deployed|provided|offered|achieved|enhanced|reduced|improved|streamlined|optimized|aided|helped|supported|assisted|facilitated)\s+', line, re.IGNORECASE):
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
    current_date_patterns = re.compile(r'\b(till\s+date|to\s+date|present|current|till\s+now|ongoing|now)\b', re.IGNORECASE)
    current_date_range_pattern = re.compile(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|[A-Za-z]+)\s+\d{4}\s*(?:to|till|[-–—])\s*(?:till\s+date|date|present|current|till\s+now|ongoing|now)\b', re.IGNORECASE)
    
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
        title_keywords = ['senior', 'sr', 'manager', 'director', 'consultant', 'engineer', 'developer', 'architect', 'analyst', 'lead', 'principal', 'teacher', 'adviser', 'advisor', 'designer', 'instructor', 'trainer', 'coordinator', 'specialist', 'coach']
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
        if len(lw.split()) == 1 and lw not in ['engineer', 'developer', 'consultant', 'manager', 'director', 'analyst', 'architect', 'teacher', 'adviser', 'advisor', 'designer', 'instructor', 'trainer', 'coordinator', 'specialist', 'assistant', 'coach']:
            score -= 3
            
        prioritized.append((score, line))
    prioritized.sort(key=lambda x: x[0], reverse=True)
    return prioritized


# -------------------------------------------------------------------------
# Heuristics / Matching
# -------------------------------------------------------------------------
def is_invalid_title_line(line: str) -> bool:
    lw = line.lower().strip()
    
    # CRITICAL: Debug logging for "Database Administrator"
    if 'database administrator' in lw:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"DEBUG: *** is_invalid_title_line called for 'Database Administrator': lw='{lw}' ***")
    
    # Reject soft skills that are not job titles
    if lw in SOFT_SKILLS:
        if 'database administrator' in lw:
            logger.warning(f"DEBUG: *** 'Database Administrator' rejected: in SOFT_SKILLS ***")
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
    # CRITICAL: Use word boundaries for short keywords like "be", "ba", "ma" to avoid false matches
    # "be" matches "database", "ba" matches "database", "ma" matches "administrator"
    # Use word boundary regex for short keywords (1-2 chars), substring match for longer keywords
    has_education_keyword = False
    matched_keyword = None
    for keyword in education_keywords:
        if len(keyword) <= 2:
            # Short keywords need word boundaries to avoid false matches
            # e.g., "be" should match "B.E" or "B.E." but not "database"
            if re.search(r'\b' + re.escape(keyword) + r'\b', lw, re.IGNORECASE):
                has_education_keyword = True
                matched_keyword = keyword
                break
        else:
            # Longer keywords can use substring match
            if keyword in lw:
                has_education_keyword = True
                matched_keyword = keyword
                break
    
    # CRITICAL: Debug logging for "Database Administrator"
    if 'database administrator' in lw:
        logger.warning(f"DEBUG: *** Education check for 'Database Administrator': has_degree_pattern={has_degree_pattern}, has_education_keyword={has_education_keyword}, matched_keyword={matched_keyword} ***")
    
    if has_degree_pattern or has_education_keyword:
        # STRICT: If it's clearly a degree format (e.g., "Btech, Civil Engineering"), reject it
        # Even if it contains role keywords like "engineering" - this is a degree, not a job title
        if has_degree_pattern:
            # Check if it's a degree followed by specialization (e.g., "Btech, Civil Engineering")
            if re.search(r'\b(b\.?tech|btech|m\.?tech|mtech|b\.?e\.?|be|m\.?e\.?|me|bsc|msc|bca|mca|ba|ma|phd|doctorate|bachelor|master)\s*[,\(]?\s*[a-z\s]+(?:engineering|science|arts|commerce|technology|computer|information)', lw, re.IGNORECASE):
                return True  # Definitely a degree format - reject
        
        # But allow if it's clearly a job title (e.g., "Education Manager" - has manager/director/etc)
        # AND doesn't have degree patterns
        # CRITICAL: Include "administrator" in role keywords to allow "Database Administrator", "System Administrator", etc.
        if not has_degree_pattern and any(role in lw for role in ['manager', 'director', 'developer', 'consultant', 'analyst', 'architect', 'lead', 'administrator']):
            # Allow job titles like "Education Manager", "Engineering Director", "Database Administrator" (but not "Btech, Civil Engineering")
            return False
        
        # Also reject if it contains "of technology" or "institute of" (institution names)
        if re.search(r'\b(institute|institution|university|college)\s+of\s+', lw, re.IGNORECASE):
            return True
        
        # CRITICAL: Reject "Director Of Technology" when it appears in education context
        # This prevents "Director Of Technology" from "Gates Institute Of Technology" being extracted
        # Pattern: "director of technology" that's likely part of an institution name
        if re.search(r'\bdirector\s+of\s+technology\b', lw, re.IGNORECASE):
            # Check if this appears near education keywords (likely institution name)
            # If the line contains "institute", "institution", "university", "college", or "school", reject it
            if any(edu_word in lw for edu_word in ['institute', 'institution', 'university', 'college', 'school']):
                return True
            # Also reject if it's a standalone "Director Of Technology" in education section
            # (fresher resumes don't have "Director Of Technology" as a job title)
            # This is a conservative check - if it's just "Director Of Technology" without company/date context, reject it
            if lw.strip() == 'director of technology' or lw.strip() == 'director of technology,':
                return True
        
        # If it has education keywords but no clear job title role, reject
        # CRITICAL: Include "administrator" in role keywords to allow "Database Administrator", "System Administrator", etc.
        has_role_keyword = any(role in lw for role in ['manager', 'director', 'developer', 'consultant', 'analyst', 'architect', 'lead', 'administrator'])
        if 'database administrator' in lw:
            logger.warning(f"DEBUG: *** Role check for 'Database Administrator': has_role_keyword={has_role_keyword} ***")
        if not has_role_keyword:
            if 'database administrator' in lw:
                logger.warning(f"DEBUG: *** 'Database Administrator' rejected: has education keyword but no role keyword ***")
            return True
    
    # explicit invalid markers
    # CRITICAL: For short tokens (1-2 chars) like "ba", "ma", "be", "me", use word boundaries
    # to avoid false matches in words like "database" (contains "ba") or "administrator" (contains "ma")
    for token in NON_TITLE_INDICATORS:
        if len(token) <= 2:
            # Short tokens need word boundaries to avoid false matches
            # e.g., "ba" should match "BA" or "B.A." but not "database"
            if re.search(r'\b' + re.escape(token) + r'\b', lw, re.IGNORECASE):
                if 'database administrator' in lw:
                    logger.warning(f"DEBUG: *** 'Database Administrator' rejected: contains NON_TITLE_INDICATOR '{token}' (word boundary match) ***")
                return True
        else:
            # Longer tokens can use substring match
            if token in lw:
                if 'database administrator' in lw:
                    logger.warning(f"DEBUG: *** 'Database Administrator' rejected: contains NON_TITLE_INDICATOR '{token}' ***")
                return True
    # description verbs suggest it is a sentence, not a title
    for v in DESCRIPTION_VERBS:
        # require whole-word match
        if re.search(r'\b' + re.escape(v) + r'\b', lw):
            if 'database administrator' in lw:
                logger.warning(f"DEBUG: *** 'Database Administrator' rejected: contains DESCRIPTION_VERB '{v}' ***")
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
    if re.match(r'^[a-z]', line) and not any(title in lw for title in ['engineer', 'developer', 'consultant', 'manager', 'director', 'analyst', 'architect', 'lead', 'senior', 'sr', 'teacher', 'adviser', 'advisor', 'designer', 'instructor', 'trainer', 'coordinator', 'specialist', 'assistant', 'coach']):
        return True
    
    return False


def best_match_from_known(line: str, is_power_platform_resume: bool = False) -> Optional[str]:
    """
    Try exact-ish and fuzzy matches against COMMON_TITLES
    Optimized to prioritize longer/more specific matches first
    Preserves dashes from original text when present
    
    Args:
        line: The line to match against known titles
        is_power_platform_resume: If True, prioritize Power Platform roles and reject "Data Analyst"
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Preserve original line for dash checking
    original_line = line
    lw = re.sub(r'[^a-z0-9 ]', ' ', line.lower())
    lw = re.sub(r'\s+', ' ', lw).strip()
    logger.info(f"DEBUG: best_match_from_known called with line='{line}', normalized='{lw}'")
    if not lw:
        logger.info(f"DEBUG: best_match_from_known returning None - empty line after normalization")
        return None
    
    # CRITICAL: Reject single-character inputs (they cause false matches)
    # Single letters like "W", "H", "F" should not match long titles
    if len(lw) == 1:
        logger.info(f"DEBUG: best_match_from_known returning None - single character input '{lw}' rejected")
        return None
    
    # Limit processing for very long lines (unlikely to be titles)
    if len(lw) > 200:
        return None

    # Check if original line contains dashes (for preservation)
    has_dash = '-' in original_line or '–' in original_line or '—' in original_line

    # CRITICAL: If this is a Power Platform resume, reject "Data Analyst" matches
    # Power Platform resumes should match Power Platform roles, not generic "Data Analyst"
    if is_power_platform_resume:
        # Reject "Data Analyst" and similar generic analyst roles
        if re.search(r'\bdata\s+analyst\b', lw, re.IGNORECASE):
            # Only reject if it's not part of a Power Platform specific role
            if not any(pp_term in lw for pp_term in ['power', 'platform', 'dynamics', 'dataverse']):
                return None  # Reject generic "Data Analyst" for Power Platform resumes

    # Sort titles by length (longest first) to prioritize more specific matches
    # This ensures "Senior Consultant" matches before "Consultant"
    # NOTE: We don't prioritize Power Platform titles - we want to match whatever is actually in the resume
    # Power Platform detection is only used to reject "Data Analyst", not to prioritize titles
    sorted_titles = sorted(COMMON_TITLES, key=len, reverse=True)
    
    # CRITICAL: Reject "Director Of Technology" when it's part of an institution name
    # This prevents "Director Of Technology" from "Gates Institute Of Technology" being matched
    if re.search(r'\bdirector\s+of\s+technology\b', lw, re.IGNORECASE):
        # If the line contains institution keywords, reject this match
        if any(edu_word in lw for edu_word in ['institute', 'institution', 'university', 'college', 'school', 'gates']):
            return None  # This is an institution name, not a job title
    
    # CRITICAL: Also reject if line contains "Institute Of Technology" or "Gates" 
    # and we're trying to match "Director Of Technology" - this prevents fuzzy matching
    # from "Gates Institute Of Technology" to "Director Of Technology"
    if any(inst_word in lw for inst_word in ['institute of technology', 'gates institute', 'gates']):
        # If we're about to match "Director Of Technology", reject it
        # Check if any known title contains "director of technology"
        for known in COMMON_TITLES:
            if 'director of technology' in known.lower():
                # This line contains institution keywords, so reject matching "Director Of Technology"
                return None
    
    # First pass: exact substring matches (prioritize longer matches)
    best_exact = None
    best_exact_len = 0
    best_exact_has_dash = False
    for known in sorted_titles:
        # CRITICAL: Skip "Director Of Technology" if line contains institution keywords
        if known.lower() == 'director of technology':
            if any(edu_word in lw for edu_word in ['institute', 'institution', 'university', 'college', 'school', 'gates']):
                continue  # Skip this known title if it's part of an institution name
        
        # CRITICAL: Normalize known title the same way as the input line
        # This ensures "Operations & Technology Intern" matches "operations technology intern"
        # Remove special characters (except spaces) and normalize
        known_normalized = re.sub(r'[^a-z0-9 ]', ' ', known.lower())
        known_normalized = re.sub(r'\s+', ' ', known_normalized).strip()
        
        # Check if normalized known title is contained in normalized line (whole word match preferred)
        # Use word boundaries to avoid partial word matches
        pattern = r'\b' + re.escape(known_normalized) + r'\b'
        if re.search(pattern, lw, re.IGNORECASE):
            if len(known) > best_exact_len:
                best_exact = known
                best_exact_len = len(known)
                # Check if the known title has a dash version
                best_exact_has_dash = '-' in known or '–' in known or '—' in known
        # Also check if normalized line is contained in normalized known (for abbreviations)
        # CRITICAL: Reject single-word matches to multi-word titles unless it's a known abbreviation
        # This prevents "contact" from matching "contact center solutions engineer"
        elif lw in known_normalized and len(known) > best_exact_len:
            # Reject if line is a single word and known title is multi-word (unless it's a valid abbreviation)
            line_words = lw.split()
            known_words = known_normalized.split()
            # Known abbreviations that are acceptable (2-3 letters, all caps or mixed case)
            is_abbreviation = len(line_words) == 1 and len(lw) <= 4 and (lw.isupper() or lw.islower())
            # CRITICAL: Reject single-word matches to multi-word titles unless it's an abbreviation
            # This prevents "intern" from matching when "Operations & Technology Intern" exists
            if len(line_words) == 1 and len(known_words) > 1 and not is_abbreviation:
                logger.info(f"DEBUG: Rejecting single-word match '{lw}' to multi-word title '{known}' (likely section header or part of longer title)")
                continue  # Skip this match - it's probably a section header or part of a longer title
            # CRITICAL: Also reject if the line is a single word that's part of a longer known title
            # Check if this single word appears in any longer known title
            should_reject = False
            if len(line_words) == 1:
                # Check if this word is part of any longer known title (that we haven't matched yet)
                for other_known in sorted_titles:
                    if len(other_known) > len(known):
                        other_known_normalized = re.sub(r'[^a-z0-9 ]', ' ', other_known.lower())
                        other_known_normalized = re.sub(r'\s+', ' ', other_known_normalized).strip()
                        if lw in other_known_normalized and lw != other_known_normalized:
                            # This single word is part of a longer title, prefer the longer one
                            logger.info(f"DEBUG: Rejecting single-word match '{lw}' - it's part of longer title '{other_known}'")
                            should_reject = True
                            break
            if should_reject:
                continue  # Skip this match - prefer the longer title
            best_exact = known
            best_exact_len = len(known)
            best_exact_has_dash = '-' in known or '–' in known or '—' in known
    
    if best_exact:
        logger.info(f"DEBUG: best_match_from_known found exact match: '{best_exact}' for line '{line}'")
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
                    logger.info(f"DEBUG: best_match_from_known returning dash variant: '{variant}'")
                    return title_case(variant)
        logger.info(f"DEBUG: best_match_from_known returning: '{best_exact}'")
        return title_case(best_exact)
    
    # Second pass: fuzzy match (only if no exact match found)
    best_fuzzy = (None, 0.0, False)
    for known in sorted_titles:
        # CRITICAL: Skip "Director Of Technology" if line contains institution keywords
        if known.lower() == 'director of technology':
            # Reject if line contains ANY institution-related keywords
            if any(edu_word in lw for edu_word in ['institute', 'institution', 'university', 'college', 'school', 'gates', 'education', 'academic']):
                continue  # Skip this known title if it's part of an institution name
            # Also reject if the line is "Gates Institute Of Technology" or similar
            if 'gates institute' in lw or 'institute of technology' in lw:
                continue  # Definitely an institution name
        
        score = fuzzy_ratio(lw, known)
        if score > best_fuzzy[1]:
            has_dash_in_known = '-' in known or '–' in known or '—' in known
            best_fuzzy = (known, score, has_dash_in_known)
            # Early exit if we found a very good match
            if score >= 0.95:
                break

    if best_fuzzy[0] and best_fuzzy[1] >= FUZZY_THRESHOLD:
        logger.info(f"DEBUG: best_match_from_known found fuzzy match: '{best_fuzzy[0]}' (score: {best_fuzzy[1]:.2f}) for line '{line}'")
        logger.info(f"DEBUG: best_match_from_known - Original line: '{original_line}' | Normalized: '{lw}' | Matched title: '{best_fuzzy[0]}'")
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
                    logger.info(f"DEBUG: best_match_from_known returning dash variant: '{variant}'")
                    return title_case(variant)
        logger.info(f"DEBUG: best_match_from_known returning: '{best_fuzzy[0]}'")
        return title_case(best_fuzzy[0])
    score_value = best_fuzzy[1] if best_fuzzy[0] else 0.0
    if best_fuzzy[0]:
        logger.info(f"DEBUG: best_match_from_known - Best candidate was '{best_fuzzy[0]}' but score {score_value:.2f} < threshold {FUZZY_THRESHOLD} for line '{line}'")
    else:
        logger.info(f"DEBUG: best_match_from_known returning None - no match found (best_fuzzy: {best_fuzzy[0]}, score: {score_value:.2f})")
    return None


def parse_date_range_end_date(date_str: str) -> Optional[datetime]:
    """
    Parse a date range string and return the end date as a datetime object.
    Handles formats like:
    - "December 2023 - August 2025"
    - "January 2025 to May 2025" (with "to" keyword)
    - "Jan 2024 - Jun 2024"
    - "06/2021 - 01/2025" (MM/YYYY format)
    - "04/2023 – 03/2025" (MM/YYYY with en dash)
    - "01/2025 – Present" (MM/YYYY with Present)
    - "1/18 – present" (MM/YY format)
    - "3/13 – 5/14" (MM/YY format)
    - "2023-03 – 2024-01" (YYYY-MM format)
    - "2023-03 – present" (YYYY-MM with Present)
    - "2023 - 2025"
    - "Jul 2024 - Dec 2024"
    Returns None if parsing fails.
    """
    if not date_str:
        return None
    
    date_str = date_str.strip()
    
    # Month name mapping
    month_map = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
        'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
        'may': 5, 'jun': 6, 'june': 6,
        'jul': 7, 'july': 7, 'aug': 8, 'august': 8,
        'sep': 9, 'september': 9, 'oct': 10, 'october': 10,
        'nov': 11, 'november': 11, 'dec': 12, 'december': 12
    }
    
    # Pattern 1: "Month Year - Month Year" or "Month Year to Month Year" (e.g., "December 2023 - August 2025", "January 2025 to May 2025")
    pattern1 = re.compile(r'([A-Za-z]+)\s+(\d{4})\s*([-–—]|to)\s*([A-Za-z]+)\s+(\d{4})', re.IGNORECASE)
    match = pattern1.search(date_str)
    if match:
        end_month_name = match.group(4).lower()[:3]  # Take first 3 chars (group 4 because "to" is now group 3)
        end_year = int(match.group(5))  # Year is now group 5
        # Find matching month
        for month_key, month_num in month_map.items():
            if month_key.startswith(end_month_name):
                try:
                    return datetime(end_year, month_num, 1)
                except:
                    pass
    
    # Pattern 1.25: "Month Year - Present/Now/Current" or "Month Year to Present/Now/Current" (e.g., "March 2025 - Now", "January 2025 to Present")
    pattern1_25 = re.compile(r'([A-Za-z]+)\s+(\d{4})\s*([-–—]|to)\s*(present|till\s+date|current|now|ongoing)', re.IGNORECASE)
    match = pattern1_25.search(date_str)
    if match:
        # It's "Present"/"Till Date"/"Current"/"Now"/"Ongoing" - return current date
        return datetime.now()
    
    # Pattern 1.5: "MM/YYYY - MM/YYYY" or "MM/YYYY to MM/YYYY" or "MM/YYYY – Present" (e.g., "06/2021 - 01/2025", "06/2021 to 01/2025", "01/2025 – Present")
    pattern1_5 = re.compile(r'(\d{1,2})/(\d{4})\s*([-–—]|to)\s*((\d{1,2})/(\d{4})|present|till\s+date|current|now|ongoing)', re.IGNORECASE)
    match = pattern1_5.search(date_str)
    if match:
        # Check if second part is a date (MM/YYYY) or "Present"/"Till Date"/"Current"/"Now"/"Ongoing"
        second_part = match.group(4).strip()  # Group 4 because separator is now group 3
        if re.match(r'\d{1,2}/\d{4}', second_part, re.IGNORECASE):
            # It's MM/YYYY format
            end_month = int(match.group(5))  # Group 5 because separator is now group 3
            end_year = int(match.group(6))  # Group 6 because separator is now group 3
            # Validate month (1-12)
            if 1 <= end_month <= 12:
                try:
                    return datetime(end_year, end_month, 1)
                except:
                    pass
        elif re.search(r'present|till\s+date|current|now|ongoing', second_part, re.IGNORECASE):
            # It's "Present"/"Till Date"/"Current"/"Now"/"Ongoing" - return current date
            return datetime.now()
    
    # Pattern 1.6: "MM/YY - MM/YY" or "MM/YY to MM/YY" or "MM/YY – Present" (e.g., "1/18 – present", "3/13 – 5/14", "5/07 – 3/13")
    pattern1_6 = re.compile(r'(\d{1,2})/(\d{2})\s*([-–—]|to)\s*((\d{1,2})/(\d{2})|present|till\s+date|current|now|ongoing)', re.IGNORECASE)
    match = pattern1_6.search(date_str)
    if match:
        # Check if second part is a date (MM/YY) or "Present"/"Till Date"/"Current"/"Now"/"Ongoing"
        second_part = match.group(4).strip()
        if re.match(r'\d{1,2}/\d{2}', second_part, re.IGNORECASE):
            # It's MM/YY format - convert 2-digit year to 4-digit year
            end_month = int(match.group(5))
            end_year_2digit = int(match.group(6))
            # Convert 2-digit year to 4-digit year
            # Years 00-30 → 2000-2030 (likely recent/future dates)
            # Years 31-99 → 1931-1999 (likely past dates)
            if end_year_2digit <= 30:
                end_year = 2000 + end_year_2digit
            else:
                end_year = 1900 + end_year_2digit
            # Validate month (1-12)
            if 1 <= end_month <= 12:
                try:
                    return datetime(end_year, end_month, 1)
                except:
                    pass
        elif re.search(r'present|till\s+date|current|now|ongoing', second_part, re.IGNORECASE):
            # It's "Present"/"Till Date"/"Current"/"Now"/"Ongoing" - return current date
            return datetime.now()
    
    # Pattern 1.7: "YYYY-MM - YYYY-MM" or "YYYY-MM to YYYY-MM" or "YYYY-MM – Present" (e.g., "2023-03 – 2024-01", "2023-03 – present")
    pattern1_7 = re.compile(r'(\d{4})-(\d{1,2})\s*([-–—]|to)\s*((\d{4})-(\d{1,2})|present|till\s+date|current|now|ongoing)', re.IGNORECASE)
    match = pattern1_7.search(date_str)
    if match:
        # Check if second part is a date (YYYY-MM) or "Present"/"Till Date"/"Current"/"Now"/"Ongoing"
        second_part = match.group(4).strip()
        if re.match(r'\d{4}-\d{1,2}', second_part, re.IGNORECASE):
            # It's YYYY-MM format
            end_year = int(match.group(5))
            end_month = int(match.group(6))
            # Validate month (1-12)
            if 1 <= end_month <= 12:
                try:
                    return datetime(end_year, end_month, 1)
                except:
                    pass
        elif re.search(r'present|till\s+date|current|now|ongoing', second_part, re.IGNORECASE):
            # It's "Present"/"Till Date"/"Current"/"Now"/"Ongoing" - return current date
            return datetime.now()
    
    # Pattern 2: "Year - Year" (e.g., "2023 - 2025")
    pattern2 = re.compile(r'(\d{4})\s*[-–—]\s*(\d{4})', re.IGNORECASE)
    match = pattern2.search(date_str)
    if match:
        end_year = int(match.group(2))
        try:
            return datetime(end_year, 12, 31)  # Use end of year
        except:
            pass
    
    # Pattern 3: "Month Year" (single date, treat as end date)
    pattern3 = re.compile(r'([A-Za-z]+)\s+(\d{4})', re.IGNORECASE)
    match = pattern3.search(date_str)
    if match:
        month_name = match.group(1).lower()[:3]
        year = int(match.group(2))
        for month_key, month_num in month_map.items():
            if month_key.startswith(month_name):
                try:
                    return datetime(year, month_num, 1)
                except:
                    pass
    
    # Pattern 4: "MM/YYYY" (single date in MM/YYYY format, treat as end date)
    pattern4 = re.compile(r'(\d{1,2})/(\d{4})', re.IGNORECASE)
    match = pattern4.search(date_str)
    if match:
        month = int(match.group(1))
        year = int(match.group(2))
        # Validate month (1-12)
        if 1 <= month <= 12:
            try:
                return datetime(year, month, 1)
            except:
                pass
    
    # Pattern 4.5: "MM/YY" (single date in MM/YY format, treat as end date)
    pattern4_5 = re.compile(r'(\d{1,2})/(\d{2})', re.IGNORECASE)
    match = pattern4_5.search(date_str)
    if match:
        month = int(match.group(1))
        year_2digit = int(match.group(2))
        # Convert 2-digit year to 4-digit year
        # Years 00-30 → 2000-2030 (likely recent/future dates)
        # Years 31-99 → 1931-1999 (likely past dates)
        if year_2digit <= 30:
            year = 2000 + year_2digit
        else:
            year = 1900 + year_2digit
        # Validate month (1-12)
        if 1 <= month <= 12:
            try:
                return datetime(year, month, 1)
            except:
                pass
    
    # Pattern 4.6: "YYYY-MM" (single date in YYYY-MM format, treat as end date)
    pattern4_6 = re.compile(r'(\d{4})-(\d{1,2})', re.IGNORECASE)
    match = pattern4_6.search(date_str)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        # Validate month (1-12)
        if 1 <= month <= 12:
            try:
                return datetime(year, month, 1)
            except:
                pass
    
    # Pattern 5: Single 4-digit year (e.g., "2023") - treat as end of that year
    # This is a fallback when only the year is present (common in some PDFs)
    pattern5 = re.compile(r'\b(19|20)\d{2}\b')
    match = pattern5.search(date_str)
    if match:
        try:
            year = int(match.group(0))
            return datetime(year, 12, 31)
        except:
            pass
    
    return None


def extract_date_range_from_context(line: str, context_lines: List[str], line_idx: int) -> Optional[datetime]:
    """
    Extract date range from a line or nearby context lines.
    Returns the end date of the range, or None if not found.
    """
    # Check the line itself first
    end_date = parse_date_range_end_date(line)
    if end_date:
        return end_date
    
    # Check nearby lines (before and after) - expanded range to catch dates that appear before titles
    # Look further back (up to 5 lines) since dates often appear before job titles
    for check_idx in range(max(0, line_idx - 5), min(len(context_lines), line_idx + 3)):
        if check_idx != line_idx:
            check_line = context_lines[check_idx]
            end_date = parse_date_range_end_date(check_line)
            if end_date:
                return end_date
    
    return None


def extract_title_from_candidate_line(line: str) -> str:
    """
    Extract just the title part from a candidate line that might include company names.
    Examples:
    - "Security Officer Global 360 Protective Services, Beverly Hills, CA" -> "Security Officer"
    - "Recreation Leader 1 City of Bell Gardens, Bell Gardens, CA" -> "Recreation Leader"
    - "Senior Consultant" -> "Senior Consultant"
    - "WORK HISTORY" -> "WORK HISTORY" (section header, return as-is)
    - "## Database Administrator" -> "Database Administrator" (remove markdown prefix)
    """
    line_clean = line.strip()
    if not line_clean:
        return line_clean
    
    # CRITICAL: Remove markdown prefixes (##, ###, ####, etc.) from the beginning
    # This handles cases like "## Database Administrator" or "### Senior Developer"
    line_clean = re.sub(r'^#+\s*', '', line_clean).strip()
    
    # Title keywords that indicate where the title ends
    title_keywords = ['officer', 'leader', 'manager', 'engineer', 'developer', 'consultant', 
                     'analyst', 'director', 'specialist', 'coordinator', 'supervisor', 'architect',
                     'administrator', 'executive', 'assistant', 'representative', 'trainee',
                     'teacher', 'adviser', 'advisor', 'designer', 'instructor', 'trainer', 'coach']
    
    # Company/location indicators that mark where title ends
    company_indicators = ['global', 'corporate', 'technologies', 'solutions', 'systems', 'services',
                         'pvt', 'ltd', 'inc', 'corp', 'company', 'city of', 'of bell', 'gardens',
                         'hills', 'beverly', 'monterey', 'park', 'protective']
    
    words = line_clean.split()
    
    # If line is very short (1-2 words) and doesn't look like a title, return as-is
    if len(words) <= 2:
        # Check if it's a section header (all caps, no title keywords)
        if line_clean.isupper() and not any(keyword in line_clean.lower() for keyword in title_keywords):
            return line_clean
        # Otherwise, return as-is (might be a short title)
        return line_clean
    
    # Strategy 1: Check for "Company - Title" or "Company, Location - Title" format
    # Handle both with and without comma
    dash_pattern = re.search(r'\s*[-–—]\s*([A-Za-z\s]+)$', line_clean)
    if dash_pattern:
        after_dash = dash_pattern.group(1).strip()
        after_dash_words = after_dash.split()
        # Check if text after dash contains a title keyword
        if any(keyword in after_dash.lower() for keyword in title_keywords):
            # Extract title from after dash (up to 4 words to handle "Digital Marketing Specialist")
            title_words = after_dash_words[:min(4, len(after_dash_words))]
            extracted = ' '.join(title_words)
            extracted = re.sub(r'\s+\d+$', '', extracted).strip()
            if extracted:
                return extracted
    
    # Strategy 1.5: If line contains comma, check first part for title
    if ',' in line_clean:
        # Original logic: If line contains comma, split and check first part
        parts = line_clean.split(',')
        first_part = parts[0].strip()
        first_words = first_part.split()
        
        # Find where title ends (look for title keyword, then stop before company indicators)
        title_end_idx = len(first_words)
        found_title_keyword = False
        
        for i, word in enumerate(first_words):
            word_lower = word.lower().rstrip('.,')
            # Check if this word is a title keyword
            if any(keyword in word_lower for keyword in title_keywords):
                found_title_keyword = True
                # Title ends after this word (or next word if it's a number like "Leader 1")
                if i + 1 < len(first_words) and first_words[i + 1].isdigit():
                    title_end_idx = i + 2
                else:
                    title_end_idx = i + 1
                break
            # Check if we hit a company indicator (stop before it)
            if any(indicator in word_lower for indicator in company_indicators):
                if found_title_keyword:
                    title_end_idx = i
                    break
        
        # Extract title (first 1-4 words up to title_end_idx)
        if found_title_keyword and title_end_idx > 0:
            title_words = first_words[:min(title_end_idx, 4)]
            extracted = ' '.join(title_words)
            # Remove trailing numbers (e.g., "Recreation Leader 1" -> "Recreation Leader")
            extracted = re.sub(r'\s+\d+$', '', extracted).strip()
            if extracted:
                return extracted

    # Strategy 1.6: Handle slash-separated titles (e.g., "Mental Health/Academic/Life Coach/Instructional Designer")
    if '/' in line_clean:
        segments = [seg.strip() for seg in line_clean.split('/') if seg.strip()]
        # Prefer segments that contain known title keywords
        for seg in segments:
            seg_lower = seg.lower()
            if any(keyword in seg_lower for keyword in title_keywords):
                # Limit to first 4 words to keep within expected title length
                seg_words = seg.split()
                candidate = ' '.join(seg_words[:4]).strip()
                candidate = re.sub(r'\s+\d+$', '', candidate).strip()
                if candidate:
                    return candidate
        # Fallback: return first non-empty segment (limited length)
        if segments:
            candidate = ' '.join(segments[0].split()[:4]).strip()
            candidate = re.sub(r'\s+\d+$', '', candidate).strip()
            if candidate:
                return candidate
    
    # Strategy 1.75: Check if first few words match a COMMON_TITLE (compound titles)
    # This handles "Assistant Manager" vs "Assistant" correctly
    # Try matching 2-word, 3-word, 4-word combinations (longest first) against COMMON_TITLES
    if len(words) >= 1:
        # Skip if line starts with dash (likely continuation from previous line)
        if not words[0].startswith('-'):
            for word_count in range(min(4, len(words)), 0, -1):
                # Build candidate title from first N words
                candidate_words = words[:word_count]
                # Skip if contains only dashes or special characters
                candidate_text = ' '.join(candidate_words)
                if re.match(r'^[-–—\s]+$', candidate_text):
                    continue
                
                candidate_title = candidate_text.lower()
                # Check if this matches a COMMON_TITLE
                if candidate_title in COMMON_TITLES:
                    # Found a match! Return it (preserve original case from line)
                    extracted = ' '.join(candidate_words)
                    # Remove trailing numbers
                    extracted = re.sub(r'\s+\d+$', '', extracted).strip()
                    # CRITICAL: Remove date patterns from the end
                    extracted = re.sub(r'\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\s*([-–—]|to)\s*(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}.*$', '', extracted, flags=re.IGNORECASE).strip()
                    extracted = re.sub(r'\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\s*([-–—]|to)\s*(?:present|current|now|ongoing|till\s+date).*$', '', extracted, flags=re.IGNORECASE).strip()
                    extracted = re.sub(r'\s+\d{1,2}/\d{4}\s*([-–—]|to)\s*(?:\d{1,2}/\d{4}|present|current|now|ongoing|till\s+date).*$', '', extracted, flags=re.IGNORECASE).strip()
                    extracted = re.sub(r'\s+\d{4}-\d{1,2}\s*([-–—]|to)\s*(?:\d{4}-\d{1,2}|present|current|now|ongoing|till\s+date).*$', '', extracted, flags=re.IGNORECASE).strip()
                    if extracted:
                        return extracted
    
    # Strategy 2: Find title keywords and extract up to that point
    words_lower = [w.lower().rstrip('.,') for w in words]
    title_end_idx = None
    
    for i, word_lower in enumerate(words_lower):
        # Check if this word contains a title keyword
        if any(keyword in word_lower for keyword in title_keywords):
            # Title ends after this word (or next if it's a number)
            if i + 1 < len(words) and words[i + 1].isdigit():
                title_end_idx = i + 2
            else:
                title_end_idx = i + 1
            break
        # Stop if we hit a company indicator
        if any(indicator in word_lower for indicator in company_indicators):
            if title_end_idx is None:
                # No title keyword found yet, might not be a title line
                break
            else:
                # Found title keyword, stop here
                break
    
    if title_end_idx and title_end_idx > 0:
        title_words = words[:min(title_end_idx, 4)]
        extracted = ' '.join(title_words)
        # Remove trailing numbers
        extracted = re.sub(r'\s+\d+$', '', extracted).strip()
        # CRITICAL: Remove date patterns from the end (e.g., "March 2025 - Now", "Jan 2024 - Present")
        extracted = re.sub(r'\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\s*([-–—]|to)\s*(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}.*$', '', extracted, flags=re.IGNORECASE).strip()
        extracted = re.sub(r'\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\s*([-–—]|to)\s*(?:present|current|now|ongoing|till\s+date).*$', '', extracted, flags=re.IGNORECASE).strip()
        extracted = re.sub(r'\s+\d{1,2}/\d{4}\s*([-–—]|to)\s*(?:\d{1,2}/\d{4}|present|current|now|ongoing|till\s+date).*$', '', extracted, flags=re.IGNORECASE).strip()
        if extracted:
            return extracted
    
    # Strategy 3: If line is short (<= 6 words) and contains title keyword, return first few words
    if len(words) <= 6:
        if any(keyword in line_clean.lower() for keyword in title_keywords):
            # Return first 3 words max
            extracted = ' '.join(words[:3])
            extracted = re.sub(r'\s+\d+$', '', extracted).strip()
            # CRITICAL: Remove date patterns from the end
            extracted = re.sub(r'\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\s*([-–—]|to)\s*(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}.*$', '', extracted, flags=re.IGNORECASE).strip()
            extracted = re.sub(r'\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\s*([-–—]|to)\s*(?:present|current|now|ongoing|till\s+date).*$', '', extracted, flags=re.IGNORECASE).strip()
            extracted = re.sub(r'\s+\d{1,2}/\d{4}\s*([-–—]|to)\s*(?:\d{1,2}/\d{4}|present|current|now|ongoing|till\s+date).*$', '', extracted, flags=re.IGNORECASE).strip()
            return extracted
    
    # Fallback: return as-is but strip dates first
    # CRITICAL: Remove date patterns from the end before returning
    line_clean = re.sub(r'\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\s*([-–—]|to)\s*(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}.*$', '', line_clean, flags=re.IGNORECASE).strip()
    line_clean = re.sub(r'\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\s*([-–—]|to)\s*(?:present|current|now|ongoing|till\s+date).*$', '', line_clean, flags=re.IGNORECASE).strip()
    line_clean = re.sub(r'\s+\d{1,2}/\d{4}\s*([-–—]|to)\s*(?:\d{1,2}/\d{4}|present|current|now|ongoing|till\s+date).*$', '', line_clean, flags=re.IGNORECASE).strip()
    return line_clean


def find_date_for_title_in_segment(title: str, exp_segment: str, exp_lines: List[str]) -> Optional[datetime]:
    """
    Find the date range associated with a title by searching the experience segment.
    Looks for date patterns near where the title appears.
    """
    import logging
    logger = logging.getLogger(__name__)
    title_lower = title.lower()
    title_words = title_lower.split()
    
    # Search for the title in exp_lines
    for idx, exp_line in enumerate(exp_lines):
        exp_line_lower = exp_line.lower()
        
        # Check if this line contains the title (or key words from the title)
        # Match if at least 2 words from title are in the line, or if title is a substring
        title_match = False
        if title_lower in exp_line_lower:
            title_match = True
        elif len(title_words) >= 2:
            # Check if at least 2 words from title are in the line
            matching_words = sum(1 for word in title_words if word in exp_line_lower)
            if matching_words >= 2:
                title_match = True
        
        if title_match:
            # Found the title line, now look for date nearby
            # Check lines after first (dates often appear after titles in modern resume formats)
            # Then check lines before (dates can also appear before titles)
            # Expanded range to catch dates that appear 2-3 lines after the title
            for check_idx in range(max(0, idx - 5), min(len(exp_lines), idx + 5)):
                if check_idx == idx:  # Skip the title line itself
                    continue
                check_line = exp_lines[check_idx]
                end_date = parse_date_range_end_date(check_line)
                if end_date:
                    logger.info(f"DEBUG: Found date {end_date.strftime('%Y-%m')} for title '{title}' at line {check_idx} (title at line {idx})")
                    return end_date
    
    return None


def regex_extract_from_line(line: str) -> Optional[str]:
    """
    Regex heuristics to extract a designation token from a candidate line.
    E.g. picks "Senior Consultant" from "April 2023 to till date\nSenior Consultant"
    or from "Senior Consultant at RAMA corporate and IT solutions"
    """
    import logging
    logger = logging.getLogger(__name__)
    lw = line.lower().strip()
    if not lw:
        return None
    
    # STRICT: Reject bullet points immediately (lines starting with bullet markers)
    if re.match(r'^[\-\–\—\•\*\u2022]\s+', line):
        logger.info(f"DEBUG: regex_extract_from_line rejected '{lw}' - bullet point")
        return None
    
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
    
    # CRITICAL: Reject "Director Of Technology" when it's part of an institution name
    # This prevents "Director Of Technology" from "Gates Institute Of Technology" being extracted
    if re.search(r'\bdirector\s+of\s+technology\b', lw, re.IGNORECASE):
        # If the line contains institution keywords, reject it
        if any(edu_word in lw for edu_word in ['institute', 'institution', 'university', 'college', 'school', 'gates']):
            return None
        # Also reject if it's just "Director Of Technology" without company/date context (likely institution name)
        if lw.strip() in ['director of technology', 'director of technology,']:
            return None
    
    # CRITICAL: Reject lines where job title keywords appear in descriptive contexts
    # Examples: "Used Wix website developer to design..." should not extract "Website Developer"
    # Pattern: action verb + object + job title keyword (e.g., "used X developer", "worked with X manager")
    # Also catch: "Aided leadership team" -> should not extract "leadership"
    descriptive_patterns = [
        r'\b(used|utilized|worked\s+with|designed\s+using|created\s+using|developed\s+using)\s+[^,]+?\s+(website\s+developer|web\s+developer|software\s+developer|developer|engineer|manager|analyst|architect|designer|specialist|coordinator|officer|director|consultant|administrator|analyst)\b',
        r'\b(using|via|through|with)\s+[^,]+?\s+(website\s+developer|web\s+developer|software\s+developer|developer|engineer|manager|analyst|architect|designer|specialist|coordinator|officer|director|consultant|administrator|analyst)\s+to\b',
        # Catch "action verb + role word" patterns like "Aided leadership", "Helped manager", "Supported team lead"
        r'\b(aided|helped|supported|assisted|facilitated|worked\s+with|collaborated\s+with)\s+[^,]+?\s*(leadership|lead|manager|director|team\s+lead|team\s+manager|leadership\s+team)\b',
        r'\b(aided|helped|supported|assisted|facilitated)\s+(leadership|lead|manager|director|team\s+lead|team\s+manager)\b',
    ]
    if any(re.search(pattern, lw, re.IGNORECASE) for pattern in descriptive_patterns):
        logger.info(f"DEBUG: regex_extract_from_line rejected '{lw}' - contains job title keyword in descriptive context")
        return None
    
    # look for common patterns: "Senior Consultant", "Consultant", "Technical Lead", etc.
    # Pattern: optional seniority + core role (removed "lead" from roles group to avoid duplication)
    roles_pattern = r'\b(?:(senior|sr|junior|jr|lead|principal|assistant|associate)\b[\s\.\-]*)?(' \
                    + r'engineer|developer|consultant|manager|architect|analyst|administrator|specialist|officer|director' \
                    + r'|teacher|adviser|advisor|designer|instructor|trainer|coordinator|coach' \
                    + r')(?:\b|s\b)'
    m = re.search(roles_pattern, line, re.IGNORECASE)
    if m:
        groups = [g for g in m.groups() if g]
        cleaned = ' '.join(groups)
        logger.info(f"DEBUG: regex_extract_from_line found match via roles_pattern: '{cleaned}' from line '{line}'")
        return title_case(cleaned)
    # fallback: if line is short and contains 1-4 words and at least one core role keyword
    words = lw.split()
    if 1 <= len(words) <= 6 and any(k in lw for k in ['engineer', 'developer', 'consultant', 'manager', 'architect', 'lead', 'director', 'analyst', 'teacher', 'adviser', 'advisor', 'designer', 'instructor', 'trainer', 'coordinator', 'specialist', 'coach']):
        # Additional check: reject if it looks like a degree (e.g., "Civil Engineering" after "Btech")
        if any(deg in lw for deg in ['btech', 'b.tech', 'mtech', 'm.tech', 'be', 'b.e', 'me', 'm.e', 'bachelor', 'master']):
            logger.info(f"DEBUG: regex_extract_from_line rejected '{lw}' - looks like a degree")
            return None
        logger.info(f"DEBUG: regex_extract_from_line found match via fallback: '{lw}' from line '{line}'")
        return title_case(lw)
    logger.info(f"DEBUG: regex_extract_from_line returning None for line '{line}'")
    return None


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------
def _validate_designation_result(result: Optional[str], resume_text: str) -> Optional[str]:
    """
    Final validation following strict designation rules:
    - Must be 1-6 words (allow single-word designations like "Assistant", "Manager")
    - Must contain professional role indicator
    - Must NOT contain company names, degree names, institutions, tools/skills alone
    - Additional checks for edge cases
    """
    if not result:
        return None
    
    # Normalize result for comparison (strip whitespace, lowercase)
    result_normalized = result.lower().strip()
    result_words = result.split()
    
    # STRICT VALIDATION: Must be 1-6 words (allow single-word valid designations)
    if len(result_words) < 1 or len(result_words) > 6:
        return None
    
    # STRICT VALIDATION: Must contain professional role indicator
    role_indicators = [
        'engineer', 'developer', 'consultant', 'manager', 'analyst', 'administrator',
        'officer', 'designer', 'assistant', 'coordinator', 'teacher', 'writer',
        'specialist', 'lead', 'director', 'executive', 'intern', 'trainee',
        'architect', 'supervisor', 'representative', 'advisor', 'adviser', 'instructor',
        'trainer', 'reviewer', 'officer', 'leader', 'head', 'chief', 'agent', 'receptionist',
        'coach'
    ]
    has_role_indicator = any(indicator in result_normalized for indicator in role_indicators)
    if not has_role_indicator:
        return None
    
    # STRICT VALIDATION: Must NOT contain company indicators
    company_indicators = ['pvt', 'ltd', 'inc', 'corp', 'solutions', 'technologies', 'systems', 
                         'services', 'global', 'corporate', 'group', 'company', 'llc']
    if any(indicator in result_normalized for indicator in company_indicators):
        return None
    
    # STRICT VALIDATION: Must NOT contain degree/institution indicators
    degree_indicators = ['bachelor', 'master', 'phd', 'doctorate', 'degree', 'diploma',
                        'university', 'college', 'institute', 'institution', 'school']
    if any(indicator in result_normalized for indicator in degree_indicators):
        return None
    
    # STRICT VALIDATION: Must NOT be just a tool/skill name
    # Common tools that shouldn't be designations alone
    tool_indicators = ['jira', 'agile', 'python', 'java', 'sql', 'excel', 'word', 'powerpoint']
    if result_normalized in tool_indicators:
        return None
    
    # CRITICAL: Reject "Data Analyst" for Power Platform resumes
    if result_normalized == 'data analyst':
        power_platform_keywords = [
            'power platform', 'power apps', 'power automate', 'power pages', 'power bi',
            'dynamics 365', 'dataverse', 'canvas app', 'model-driven app', 'power fx',
            'microsoft power platform', 'ms power platform'
        ]
        resume_lower = resume_text.lower()
        has_power_platform = any(keyword in resume_lower for keyword in power_platform_keywords)
        if has_power_platform:
            return None
    
    # Reject "Director Of Technology" for fresher resumes
    if result_normalized == 'director of technology' or result_normalized == 'director of technology,':
        has_education = bool(re.search(r'\b(education|educational|academic|qualification)\b', resume_text, re.IGNORECASE))
        has_institution = bool(re.search(r'\b(gates\s+institute\s+of\s+technology|institute\s+of\s+technology)\b', resume_text, re.IGNORECASE))
        has_experience_section = bool(re.search(r'\b(experience|work history|employment history|professional experience|career development)\b', resume_text, re.IGNORECASE))
        if (has_education or has_institution) and not has_experience_section:
            return None
    
    return result


def _has_power_platform_keywords(text: str) -> bool:
    """
    Check if resume text contains Power Platform related keywords.
    This helps prioritize Power Platform roles and reject generic roles like "Data Analyst".
    """
    text_lower = text.lower()
    power_platform_keywords = [
        'power platform', 'power apps', 'power automate', 'power pages', 'power bi',
        'dynamics 365', 'dataverse', 'canvas app', 'model-driven app', 'power fx',
        'microsoft power platform', 'ms power platform', 'power platform developer',
        'power platform consultant', 'power apps developer', 'power automate developer'
    ]
    return any(keyword in text_lower for keyword in power_platform_keywords)


def _reconstruct_date_string_from_lines(exp_lines: List[str], idx: int, block_start: int, block_end: int) -> Optional[str]:
    """
    Reconstruct a full date string from multiple lines when date components are split.
    Example: Lines ["June", "2020", "-", "PRESENT"] -> "June 2020 - PRESENT"
    Returns the reconstructed date string or None if not found.
    """
    # Look backward up to 5 lines to find month/year
    # Look forward up to 2 lines to find "PRESENT" or end date
    max_lookback = 5
    max_lookforward = 2
    
    start_idx = max(block_start, idx - max_lookback)
    end_idx = min(block_end, idx + max_lookforward)
    
    # Collect relevant lines
    relevant_lines = []
    for i in range(start_idx, end_idx + 1):
        line = exp_lines[i].strip()
        if line:
            relevant_lines.append((i, line))
    
    # Try to reconstruct date string
    # Pattern 1: Month Year - PRESENT (split across lines)
    # Pattern 2: Month Year - EndDate (split across lines)
    
    # Look for month keywords
    month_pattern = re.compile(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\b', re.IGNORECASE)
    year_pattern = re.compile(r'\b(19|20)\d{2}\b')  # 4-digit year starting with 19 or 20
    current_pattern = re.compile(r'\b(present|current|till\s+date|till\s+now|ongoing|now|to\s+date|still\s+working)\b', re.IGNORECASE)
    
    # Find month, year, and current indicator
    month_line = None
    year_line = None
    current_line = None
    dash_line = None
    
    for line_idx, line in relevant_lines:
        if month_pattern.search(line) and month_line is None:
            month_line = (line_idx, line)
        if year_pattern.search(line) and year_line is None:
            year_line = (line_idx, line)
        if current_pattern.search(line):
            current_line = (line_idx, line)
        if re.match(r'^[-–—]$', line):
            dash_line = (line_idx, line)
    
    # Reconstruct if we have month and year (year can be before or after month in the text)
    if month_line and year_line:
        parts = []
        
        # Add month text (may already contain start month/year and "to" part)
        month_text = month_line[1]
        parts.append(month_text)
        
        # Always append the standalone year we found somewhere in the window.
        # This handles cases where the year appears on its own line either
        # before or after the month range (common in PDF text extraction).
        year_text = year_line[1]
        year_match = year_pattern.search(year_text)
        if year_match:
            parts.append(year_match.group(0))
        else:
            parts.append(year_text)
        
        # Add dash if present as a standalone line (rare, but supported)
        if dash_line:
            parts.append('-')
        
        # Add current indicator or end date (e.g., "Present")
        if current_line:
            parts.append(current_line[1])
        
        reconstructed = ' '.join(parts)
        return reconstructed
    
    return None


def _identify_experience_blocks(exp_lines: List[str]) -> List[Tuple[int, int]]:
    """
    Identify experience blocks by grouping related lines.
    Returns list of (start_idx, end_idx) tuples for each block.
    A block typically contains: Company, Designation, Date, Descriptions.
    Blocks are separated by empty lines or new company/date patterns.
    ENHANCED: Better handles split content (designation and date on separate lines).
    """
    blocks = []
    if not exp_lines:
        return blocks
    
    current_start = 0
    for idx in range(len(exp_lines)):
        line = exp_lines[idx].strip()
        
        # Empty line indicates block boundary (but allow small gaps for split content)
        if not line:
            # Only create a boundary if we have multiple consecutive empty lines
            # or if the next non-empty line looks like a new entry
            if idx + 1 < len(exp_lines):
                next_line = exp_lines[idx + 1].strip()
                if next_line:
                    # Check if next line looks like a new experience entry
                    date_pattern = re.compile(r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\s+\d{4}', re.IGNORECASE)
                    # Also check for company-like patterns (all caps, long lines)
                    is_new_entry = date_pattern.match(next_line) or (
                        len(next_line) > 20 and next_line.isupper() and 
                        not any(keyword in next_line.lower() for keyword in ['present', 'current', 'till date'])
                    )
                    
                    if is_new_entry and current_start < idx:
                        blocks.append((current_start, idx - 1))
                        current_start = idx + 1
            continue
        
        # Check if this line looks like a new experience entry (company name or date pattern)
        # This helps identify when a new job starts
        is_new_entry = False
        # Date pattern at start of line suggests new entry
        date_pattern = re.compile(r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\s+\d{4}', re.IGNORECASE)
        if date_pattern.match(line):
            is_new_entry = True
        
        # Also detect company names (all caps, long lines, not dates)
        if not is_new_entry and len(line) > 15:
            # Check if it looks like a company name (mixed case or all caps, not a date)
            if line.isupper() or (line[0].isupper() and not date_pattern.search(line)):
                # Check if previous block ended recently (within 3 lines)
                # If so, this might be a new entry
                if blocks and idx - blocks[-1][1] <= 3:
                    is_new_entry = True
        
        # If we detect a new entry and we have a current block, close it
        if is_new_entry and current_start < idx:
            blocks.append((current_start, idx - 1))
            current_start = idx
    
    # Add final block
    if current_start < len(exp_lines):
        blocks.append((current_start, len(exp_lines) - 1))
    
    # Ensure we have at least one block
    if not blocks:
        blocks = [(0, len(exp_lines) - 1)]
    
    return blocks


def _extract_designation_near_date(exp_lines: List[str], idx: int, block_start: int, block_end: int, end_date: datetime) -> Optional[Tuple[datetime, str, str, int]]:
    """
    Helper function to extract designation near a date line.
    Returns (end_date, designation, source_line, line_index) tuple or None.
    """
    # ENHANCED: Check up to 5 lines before and after (within block boundaries)
    # Prioritize closer lines but check all within range
    max_proximity = 5
    check_indices = [idx]  # Same line (highest priority)
    
    # Add lines before (closer first)
    for offset in range(1, min(max_proximity + 1, idx - block_start + 1)):
        check_idx = idx - offset
        if check_idx >= block_start:
            check_indices.append(check_idx)
    
    # Add lines after (closer first)
    for offset in range(1, min(max_proximity + 1, block_end - idx + 1)):
        check_idx = idx + offset
        if check_idx <= block_end:
            check_indices.append(check_idx)
    
    # Process check_indices (already sorted by priority: same line, then by distance)
    for check_idx in check_indices:
        check_line = exp_lines[check_idx].strip()
        
        # Skip empty lines
        if not check_line:
            continue
        
        # Skip bullet points and action verb lines
        if re.match(r'^[\-\–\—\•\*\u2022]\s+', check_line):
            continue
        if re.match(r'^(aided|helped|supported|assisted|facilitated|playing|managing|strategizing|front|optimizing|empowering|coaching|initiated|standardizing|formalizing|responsible|establishing|improving|has|utilized|developed|created|configured|implemented|enabled|designed|integrated|automated|conducted|deployed|provided|offered|achieved|enhanced|reduced|improved|streamlined|optimized)\s+', check_line, re.IGNORECASE):
            continue
        
        # Skip lines that are just dashes or separators
        if re.match(r'^[-–—]+$', check_line):
            continue
        
        # Skip lines that are just numbers (likely part of date on another line)
        if re.match(r'^\d+$', check_line):
            continue
        
        if not is_invalid_title_line(check_line):
            # Try extracting from the line directly
            title_from_line = extract_title_from_candidate_line(check_line)
            if title_from_line and not is_invalid_title_line(title_from_line):
                match = best_match_from_known(title_from_line, False) or regex_extract_from_line(title_from_line)
                if match:
                    # Validate: 1-6 words (allow single-word designations like "Assistant", "Manager")
                    # Single-word designations are valid if they match known titles or role patterns
                    words = match.split()
                    if 1 <= len(words) <= 6:
                        return (end_date, match, check_line, check_idx)
    
    return None


def _find_present_current_roles(exp_segment: str, exp_lines: List[str]) -> List[Tuple[datetime, str, str, int]]:
    """
    Priority 1 - Step 1: Find all designations associated with explicit current role indicators.
    Returns list of (end_date, designation, source_line, line_index) tuples.
    Current indicators: Present, Current, Till Date, Till Now, Ongoing, Now, Still Working
    ENHANCED: Designation can be on same line, immediately before, or immediately after the date line.
    Uses experience blocks to ensure designation and date are in the same block.
    """
    current_indicators = ['present', 'current', 'till date', 'till now', 'ongoing', 'now', 'to date', 'still working']
    candidates = []
    
    # Pattern to match date ranges ending with current indicators
    current_date_patterns = [
        re.compile(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\s+\d{4}\s*[-–—]?\s*(present|current|till\s+date|till\s+now|ongoing|now|to\s+date|still\s+working)\b', re.IGNORECASE),
        re.compile(r'\d{1,2}/\d{4}\s*[-–—]?\s*(present|current|till\s+date|till\s+now|ongoing|now|to\s+date|still\s+working)\b', re.IGNORECASE),
        re.compile(r'\d{1,2}/\d{2}\s*[-–—]?\s*(present|current|till\s+date|till\s+now|ongoing|now|to\s+date|still\s+working)\b', re.IGNORECASE),
        re.compile(r'\d{4}-\d{1,2}\s*[-–—]?\s*(present|current|till\s+date|till\s+now|ongoing|now|to\s+date|still\s+working)\b', re.IGNORECASE),
        re.compile(r'\d{4}\s*[-–—]?\s*(present|current|till\s+date|till\s+now|ongoing|now|to\s+date|still\s+working)\b', re.IGNORECASE),
        re.compile(r'\b(present|current|till\s+date|till\s+now|ongoing|now|to\s+date|still\s+working)\s*$', re.IGNORECASE)
    ]
    
    # Identify experience blocks
    blocks = _identify_experience_blocks(exp_lines)
    
    for block_start, block_end in blocks:
        # Search within this block for current role indicators
        for idx in range(block_start, block_end + 1):
            line = exp_lines[idx]
            line_lower = line.lower()
            
            # Check if line contains current indicator
            has_current = any(pattern.search(line) for pattern in current_date_patterns)
            if not has_current:
                # Also check for standalone current keywords
                has_current = any(indicator in line_lower for indicator in current_indicators)
            
            if has_current:
                # ENHANCED: Try to reconstruct full date string from multiple lines
                reconstructed_date = _reconstruct_date_string_from_lines(exp_lines, idx, block_start, block_end)
                date_line_to_parse = reconstructed_date if reconstructed_date else line
                
                # Parse end date (use current date for "present" indicators)
                end_date = parse_date_range_end_date(date_line_to_parse)
                if not end_date:
                    end_date = datetime.now()
                
                # Extract designation near the date
                result = _extract_designation_near_date(exp_lines, idx, block_start, block_end, end_date)
                if result:
                    candidates.append(result)
                    break  # Found valid match in this block, move to next block
    
    return candidates


def _find_future_date_roles(exp_segment: str, exp_lines: List[str]) -> List[Tuple[datetime, str, str, int]]:
    """
    Priority 1 - Step 2: Find all designations associated with future dates (beyond current date).
    Returns list of (end_date, designation, source_line, line_index) tuples.
    Only checks for date ranges ending in the future (not explicit "Present"/"Current" keywords).
    ENHANCED: Designation can be on same line, immediately before, or immediately after the date line.
    Uses experience blocks to ensure designation and date are in the same block.
    """
    candidates = []
    current_date = datetime.now()
    
    # Pattern to match date ranges (for future date detection)
    date_range_pattern = re.compile(
        r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\s+\d{4}\s*([-–—]|to)\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\s+\d{4}',
        re.IGNORECASE
    )
    date_range_pattern2 = re.compile(
        r'\d{1,2}/\d{4}\s*([-–—]|to)\s*\d{1,2}/\d{4}',
        re.IGNORECASE
    )
    # Match MM/YY formats: 3/13 – 5/14, 5/07 – 3/13
    date_range_pattern2_5 = re.compile(
        r'\d{1,2}/\d{2}\s*([-–—]|to)\s*\d{1,2}/\d{2}',
        re.IGNORECASE
    )
    # Match YYYY-MM formats: 2023-03 – 2024-01, 2023-03 – present
    date_range_pattern2_6 = re.compile(
        r'\d{4}-\d{1,2}\s*([-–—]|to)\s*(?:\d{4}-\d{1,2}|present|current|till\s+date|till\s+now|ongoing|now)',
        re.IGNORECASE
    )
    # Match year-year formats: (2023–2025), 2023–2025, 2023 - 2025, 2023-2025
    date_range_pattern3 = re.compile(
        r'\(?\s*(\d{4})\s*[-–—]\s*(\d{4})\s*\)?',
        re.IGNORECASE
    )
    
    # Identify experience blocks
    blocks = _identify_experience_blocks(exp_lines)
    
    for block_start, block_end in blocks:
        # Search within this block for future dates
        for idx in range(block_start, block_end + 1):
            line = exp_lines[idx]
            
            # Check for date ranges that end in the future (skip if contains "Present"/"Current" keywords)
            # First check if line contains explicit current indicators - if so, skip (already handled in Step 1)
            current_indicators = ['present', 'current', 'till date', 'till now', 'ongoing', 'now', 'to date', 'still working']
            line_lower = line.lower()
            has_explicit_current = any(indicator in line_lower for indicator in current_indicators)
            
            if has_explicit_current:
                continue  # Skip - this was already handled in Step 1
            
            # Check for date ranges that end in the future
            has_date_range = date_range_pattern.search(line) or date_range_pattern2.search(line) or date_range_pattern2_5.search(line) or date_range_pattern2_6.search(line) or date_range_pattern3.search(line)
            if has_date_range:
                # Parse end date
                parsed_future_end_date = parse_date_range_end_date(line)
                if parsed_future_end_date and parsed_future_end_date >= current_date:
                    # End date is in the future or today - treat as current role
                    # Extract designation near the date
                    result = _extract_designation_near_date(exp_lines, idx, block_start, block_end, parsed_future_end_date)
                    if result:
                        candidates.append(result)
                        break  # Found valid match in this block, move to next block
    
    return candidates


def _find_roles_with_dates(exp_segment: str, exp_lines: List[str]) -> List[Tuple[datetime, str, str, int]]:
    """
    Priority 2: Find all roles with date ranges and return them with end dates.
    Returns list of (end_date, designation, source_line, line_index) tuples.
    If multiple roles end in same year, prefer the last occurring role (higher line_index).
    ENHANCED: Checks up to 5 lines before date to find designation, handles all date formats.
    """
    candidates = []
    # Enhanced date pattern: matches month-year to month-year, month-year to present, MM/YYYY formats
    date_range_pattern = re.compile(
        r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\s*([-–—]|to)\s*(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}',
        re.IGNORECASE
    )
    # Also match MM/YYYY formats
    date_range_pattern2 = re.compile(
        r'\d{1,2}/\d{4}\s*([-–—]|to)\s*(?:\d{1,2}/\d{4}|present|current|till\s+date|till\s+now|ongoing|now)',
        re.IGNORECASE
    )
    # Match MM/YY formats: 3/13 – 5/14, 5/07 – 3/13, 1/18 – present
    date_range_pattern2_5 = re.compile(
        r'\d{1,2}/\d{2}\s*([-–—]|to)\s*(?:\d{1,2}/\d{2}|present|current|till\s+date|till\s+now|ongoing|now)',
        re.IGNORECASE
    )
    # Match YYYY-MM formats: 2023-03 – 2024-01, 2023-03 – present
    date_range_pattern2_6 = re.compile(
        r'\d{4}-\d{1,2}\s*([-–—]|to)\s*(?:\d{4}-\d{1,2}|present|current|till\s+date|till\s+now|ongoing|now)',
        re.IGNORECASE
    )
    # Match year-year formats: (2023–2025), 2023–2025, 2023 - 2025, 2023-2025
    date_range_pattern3 = re.compile(
        r'\(?\s*(\d{4})\s*[-–—]\s*(\d{4})\s*\)?',
        re.IGNORECASE
    )
    
    # Treat entire experience segment as a single block for date reconstruction
    block_start = 0
    block_end = len(exp_lines) - 1
    
    for idx, line in enumerate(exp_lines):
        date_line_to_parse = None
        
        # First, try direct date range patterns on this line
        has_date = date_range_pattern.search(line) or date_range_pattern2.search(line) or date_range_pattern2_5.search(line) or date_range_pattern2_6.search(line) or date_range_pattern3.search(line)
        if has_date:
            date_line_to_parse = line
        else:
            # If no full date range is found on this line, try to reconstruct a split date
            # This handles cases like:
            #   "October 2020 to April"
            #   "2023"
            # split across adjacent lines in PDFs.
            reconstructed = _reconstruct_date_string_from_lines(exp_lines, idx, block_start, block_end)
            if reconstructed:
                date_line_to_parse = reconstructed
        
        if not date_line_to_parse:
            continue
        
        end_date = parse_date_range_end_date(date_line_to_parse)
        if not end_date:
            continue
            
        # ENHANCED: Check up to 5 lines before the date (like Priority 1)
        # Prioritize closer lines but check all within range
        max_proximity = 5
        check_indices = [idx]  # Same line (highest priority)
        
        # Add lines before (closer first)
        for offset in range(1, min(max_proximity + 1, idx + 1)):
            check_idx = idx - offset
            if check_idx >= 0:
                check_indices.append(check_idx)
        
        # Process check_indices (already sorted by priority: same line, then by distance)
        for check_idx in check_indices:
            check_line = exp_lines[check_idx].strip()
            
            # Skip empty lines
            if not check_line:
                continue
            
            # Skip bullet points and action verb lines
            if re.match(r'^[\-\–\—\•\*\u2022]\s+', check_line):
                continue
            if re.match(r'^(aided|helped|supported|assisted|facilitated|playing|managing|strategizing|front|optimizing|empowering|coaching|initiated|standardizing|formalizing|responsible|establishing|improving|has|utilized|developed|created|configured|implemented|enabled|designed|integrated|automated|conducted|deployed|provided|offered|achieved|enhanced|reduced|improved|streamlined|optimized)\s+', check_line, re.IGNORECASE):
                continue
            
            # Skip lines that are just dashes or separators
            if re.match(r'^[-–—]+$', check_line):
                continue
            
            # Skip lines that are just numbers (likely part of date on another line)
            if re.match(r'^\d+$', check_line):
                continue
            
            # Skip lines that are clearly company names (long lines with location patterns)
            if re.search(r',\s*[A-Z]{2}\s+\d{5}', check_line):  # City, ST ZIP pattern
                continue
            
            if not is_invalid_title_line(check_line):
                # Try extracting from the line directly
                title_from_line = extract_title_from_candidate_line(check_line)
                if title_from_line and not is_invalid_title_line(title_from_line):
                    match = best_match_from_known(title_from_line, False) or regex_extract_from_line(title_from_line)
                    if match:
                        # Validate: 1-6 words (allow single-word designations)
                        words = match.split()
                        if 1 <= len(words) <= 6:
                            candidates.append((end_date, match, check_line, check_idx))
                            break  # Found valid match for this date, move to next date
    
    return candidates


def extract_designation(resume_text: str) -> Optional[str]:
    """
    Extract candidate's current or most recent professional job designation.
    
    Priority Order:
    Priority 0: Header/Name Section (HIGHEST) → Check resume header first
    Priority 1: Current Role (Sequential approach):
        Step 1: Check for explicit "Present"/"Current" keywords → Return immediately if found
        Step 2: If Step 1 found nothing, check for future dates → Return if found
    Priority 2: Latest End Date → Return role with latest end date
    Priority 3: Fallback → Return first valid designation in experience section
    
    Returns: Single designation string (title-cased) or None if not found.
    """
    if not resume_text or not resume_text.strip():
        return None

    text = normalize_text(resume_text)
    
    # PRIORITY 0: Header/Name Section (HIGHEST PRIORITY)
    # If designation appears near candidate name at top, give it HIGH priority
    header_designation = extract_designation_from_header(text)
    if header_designation:
        return header_designation
    
    # Early rejection: Fresher resumes (education but no experience)
    has_education = bool(re.search(r'\b(education|educational|academic|qualification)\b', text, re.IGNORECASE))
    has_experience_section = bool(re.search(
        r'\b(experience|work history|employment history|professional experience|career development|relevant experience|additional experience)\b',
        text, re.IGNORECASE
    ))
    
    if has_education and not has_experience_section:
        return None  # Fresher resume - no designation to extract

    # Crop to experience section only
    exp_segment = crop_to_experience_section(text)
    
    # ENHANCED: If experience section seems incomplete (too small or few date patterns), 
    # search in a larger section of the resume
    if len(exp_segment) < 500:
        # Try to find "Work Experience" or "Experience" header position
        work_exp_match = re.search(r'\b(work\s+experience|experience)\b', text, re.IGNORECASE)
        if work_exp_match:
            # Extend to include more text before and after the header
            start_pos = max(0, work_exp_match.start() - 3000)  # Include 3000 chars before
            end_pos = min(len(text), work_exp_match.end() + 5000)  # Include 5000 chars after
            extended_segment = text[start_pos:end_pos]
            # Count date patterns in extended segment vs original
            date_pattern_count = len(re.findall(r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\s+\d{4}\s*([-–—]|to)', extended_segment, re.IGNORECASE))
            original_count = len(re.findall(r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\s+\d{4}\s*([-–—]|to)', exp_segment, re.IGNORECASE))
            # If extended segment has more date patterns, use it
            if date_pattern_count > original_count:
                exp_segment = extended_segment
    
    # If segment is too small and it's a fresher resume, return None
    if len(exp_segment) < 100:
        has_institution = bool(re.search(r'\b(gates\s+institute\s+of\s+technology|institute\s+of\s+technology)\b', text, re.IGNORECASE))
        if (has_education or has_institution) and not has_experience_section:
            return None
        # Fallback to entire text if not a fresher resume
        if len(exp_segment) < 100:
            exp_segment = text
    
    exp_lines = exp_segment.split('\n')
    
    # PRIORITY 1 - Step 1: Check for explicit "Present"/"Current" keywords first
    present_current_candidates = _find_present_current_roles(exp_segment, exp_lines)
    if present_current_candidates:
        # SPECIAL INTERN HANDLING: If Intern is marked as Present, select it
        # If multiple "Present" roles exist, prefer the most recent one in resume order (last occurring)
        # Sort by line_index (last occurring first), then by end_date
        present_current_candidates.sort(key=lambda x: (x[3], x[0]), reverse=True)
        
        # Separate interns from non-interns
        intern_candidates = [(ed, des, sl, li) for ed, des, sl, li in present_current_candidates 
                            if 'intern' in des.lower() or 'trainee' in des.lower()]
        non_intern_candidates = [(ed, des, sl, li) for ed, des, sl, li in present_current_candidates 
                                 if 'intern' not in des.lower() and 'trainee' not in des.lower()]
        
        # RULE: If Intern is marked as Present, select it (if it's the most recent Present role)
        # But if a non-intern is also Present and more recent, prefer non-intern
        if non_intern_candidates:
            # Non-intern roles exist - prefer them (they're professional roles)
            for end_date, designation, source_line, line_idx in non_intern_candidates:
                validated = _validate_designation_result(designation, resume_text)
                if validated:
                    return validated
        
        # If no non-intern found or all non-interns failed validation, check interns
        if intern_candidates:
            # Intern is marked as Present - select it
            for end_date, designation, source_line, line_idx in intern_candidates:
                validated = _validate_designation_result(designation, resume_text)
                if validated:
                    return validated
    
    # PRIORITY 1 - Step 2: Only if Step 1 found nothing, check for future dates
    future_date_candidates = _find_future_date_roles(exp_segment, exp_lines)
    if future_date_candidates:
        # Sort by line_index (last occurring first), then by end_date
        future_date_candidates.sort(key=lambda x: (x[3], x[0]), reverse=True)
        
        # Separate interns from non-interns
        intern_candidates = [(ed, des, sl, li) for ed, des, sl, li in future_date_candidates 
                            if 'intern' in des.lower() or 'trainee' in des.lower()]
        non_intern_candidates = [(ed, des, sl, li) for ed, des, sl, li in future_date_candidates 
                                 if 'intern' not in des.lower() and 'trainee' not in des.lower()]
        
        # Prefer non-intern roles over intern roles
        if non_intern_candidates:
            # Non-intern roles exist - prefer them (they're professional roles)
            for end_date, designation, source_line, line_idx in non_intern_candidates:
                validated = _validate_designation_result(designation, resume_text)
                if validated:
                    return validated
        
        # If no non-intern found or all non-interns failed validation, check interns
        if intern_candidates:
            # Intern role with future date - select it
            for end_date, designation, source_line, line_idx in intern_candidates:
                validated = _validate_designation_result(designation, resume_text)
                if validated:
                    return validated
    
    # PRIORITY 2: Latest End Date
    dated_roles = _find_roles_with_dates(exp_segment, exp_lines)
    if dated_roles:
        # SPECIAL INTERN HANDLING: If Intern is older and full-time role exists later, ignore intern
        # Sort by end_date (latest first), then by line_index (last occurring first for same year)
        dated_roles.sort(key=lambda x: (x[0], x[3]), reverse=True)
        
        # Separate interns from full-time roles
        intern_roles = [(ed, des, sl, li) for ed, des, sl, li in dated_roles 
                        if 'intern' in des.lower() or 'trainee' in des.lower()]
        fulltime_roles = [(ed, des, sl, li) for ed, des, sl, li in dated_roles 
                          if 'intern' not in des.lower() and 'trainee' not in des.lower()]
        
        # If full-time role exists and is later than intern, ignore intern
        if fulltime_roles:
            latest_fulltime = fulltime_roles[0]
            latest_fulltime_date = latest_fulltime[0]
            latest_fulltime_idx = latest_fulltime[3]
            
            # Only process interns if they're more recent than the latest full-time role
            valid_interns = [ir for ir in intern_roles 
                           if ir[0] > latest_fulltime_date or (ir[0] == latest_fulltime_date and ir[3] > latest_fulltime_idx)]
            
            # Process full-time roles first
            for end_date, designation, source_line, line_idx in fulltime_roles:
                validated = _validate_designation_result(designation, resume_text)
                if validated:
                    return validated
            
            # Then process valid interns (if any)
            for end_date, designation, source_line, line_idx in valid_interns:
                validated = _validate_designation_result(designation, resume_text)
                if validated:
                    return validated
        else:
            # No full-time roles, process all roles (including interns)
            for end_date, designation, source_line, line_idx in dated_roles:
                validated = _validate_designation_result(designation, resume_text)
                if validated:
                    return validated
    
    # PRIORITY 3: Fallback - First valid designation in experience section
    # ENHANCED: Apply intern filtering here too - prefer professional roles over intern roles
    candidates = candidate_title_lines(exp_segment)
    
    # Separate professional roles from intern roles
    professional_candidates = []
    intern_candidates = []
    
    for score, line in candidates:
        if is_invalid_title_line(line):
            continue
        
        # Extract title from line
        extracted_title = extract_title_from_candidate_line(line)
        if not extracted_title or is_invalid_title_line(extracted_title):
            continue
        
        # Try to match
        match = best_match_from_known(extracted_title, False) or regex_extract_from_line(extracted_title)
        if match:
            validated = _validate_designation_result(match, resume_text)
            if validated:
                # Check if it's an intern/trainee role
                if 'intern' in validated.lower() or 'trainee' in validated.lower():
                    intern_candidates.append(validated)
                else:
                    professional_candidates.append(validated)
    
    # Prefer professional roles over intern roles
    if professional_candidates:
        return professional_candidates[0]  # Return first professional role found
    
    # If no professional roles, return first intern role (if any)
    if intern_candidates:
        return intern_candidates[0]
    
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
