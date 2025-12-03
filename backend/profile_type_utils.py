"""
Utility helpers to derive and reuse candidate profile types (Java, .Net, SAP, etc.).

The goal is to keep a single source of truth for profile type detection so that
resume parsing, SQL filtering, and search/ranking code stay in sync.
"""

import logging
import re
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple, Dict, Set, NamedTuple, Any

logger = logging.getLogger(__name__)

# Legacy rules for backward compatibility (simple set-based)
PROFILE_TYPE_RULES = [
    ("Java", {
        "java", "jdk", "jvm", "jre", "spring", "spring boot", "spring mvc", "spring cloud",
        "hibernate", "jpa", "j2ee", "jakarta ee", "servlet", "jsp", "jsf", "struts",
        "microservices", "rest api", "maven", "gradle", "ant", "jboss", "weblogic", "tomcat"
    }),
    (".Net", {
        ".net", "dotnet", "c#", "csharp", "asp.net", "asp.net core", ".net core", ".net framework",
        "entity framework", "ef core", "wpf", "winforms", "web api", "linq", "xamarin", "blazor",
        "ado.net", "wcf", "mvc", "razor", "signalr"
    }),
    ("Python", {
        "python", "django", "flask", "fastapi", "tornado", "pyramid", "bottle",
        "pandas", "numpy", "scipy", "matplotlib", "seaborn", "sqlalchemy", "celery",
        "redis", "asyncio", "pytest", "jupyter", "scikit-learn"
    }),
    ("JavaScript", {
        "javascript", "js", "node", "node.js", "react", "angular", "vue", "typescript",
        "es6", "jquery", "express", "next.js", "nuxt.js", "nuxt", "webpack", "babel",
        "gulp", "grunt", "npm", "yarn", "redux", "mobx"
    }),
    ("Full Stack", {
        "full stack", "fullstack", "mern", "mean", "mevn", "lamp", "lemp",
        "next.js", "nuxt", "react", "node.js", "express", "mongodb", "postgresql", "mysql",
        "frontend", "backend", "full stack developer"
    }),
    ("DevOps", {
        "devops", "ci/cd", "cicd", "jenkins", "gitlab ci", "github actions", "azure devops",
        "docker", "kubernetes", "k8s", "helm", "terraform", "ansible", "chef", "puppet",
        "monitoring", "prometheus", "grafana", "elk", "splunk", "nagios"
    }),
    ("Cloud / Infra", {
        "aws", "azure", "gcp", "google cloud", "cloud architect", "cloud engineer",
        "ec2", "s3", "lambda", "iam", "eks", "aks", "gke", "cloudformation", "vmware",
        "vsphere", "openstack", "cloud infrastructure"
    }),
    ("Data Engineering", {
        "data engineer", "data engineering", "etl", "elt", "airflow", "luigi",
        "snowflake", "spark", "pyspark", "hadoop", "hdfs", "databricks", "kafka",
        "flume", "sqoop", "redshift", "bigquery", "data pipeline", "data warehouse"
    }),
    ("Data Science", {
        "data science", "machine learning", "deep learning", "ml engineer", "ai engineer",
        "artificial intelligence", "llm", "nlp", "computer vision", "tensorflow", "pytorch",
        "scikit-learn", "keras", "neural network", "data scientist"
    }),
    ("Business Intelligence (BI)", {
        "power bi", "tableau", "qlik", "looker", "ssis", "ssrs", "business intelligence",
        "bi", "data visualization", "dashboard", "cognos", "microstrategy", "spotfire"
    }),
    ("Testing / QA", {
        "qa", "quality assurance", "manual testing", "automation testing", "test automation",
        "selenium", "cypress", "playwright", "junit", "testng", "pytest", "mocha", "jest",
        "postman", "api testing", "performance testing", "jmeter", "load testing"
    }),
    ("SAP", {
        "sap", "abap", "hana", "sap hana", "s/4hana", "successfactors", "ariba", "bw",
        "fiori", "sap mm", "sap sd", "sap fico", "sap basis", "sap pi", "sap po"
    }),
    ("ERP", {
        "erp", "oracle apps", "oracle e-business", "oracle fusion", "d365", "dynamics 365",
        "dynamics", "business central", "netsuite", "workday", "odoo", "peopleSoft", "sage"
    }),
    ("Microsoft Power Platform", {
        "power platform", "power apps", "power automate", "power bi", "power virtual agents",
        "dataverse", "power fx", "canvas app", "model-driven app", "power pages"
    }),
    ("RPA", {
        "rpa", "robotic process automation", "ui path", "uipath", "automation anywhere",
        "blue prism", "rpa developer", "process automation", "workfusion"
    }),
    ("Cyber Security", {
        "cyber security", "cybersecurity", "information security", "soc", "security operations center",
        "siem", "penetration testing", "pen testing", "ethical hacking", "vapt", "vulnerability assessment",
        "security analyst", "security engineer", "threat hunting", "incident response"
    }),
    ("Mobile Development", {
        "android", "ios", "kotlin", "swift", "flutter", "react native", "xamarin",
        "mobile app", "ios development", "android development", "ionic", "cordova"
    }),
    ("Salesforce", {
        "salesforce", "sfdc", "apex", "visualforce", "lightning", "lwc", "lightning web components",
        "soql", "sosl", "salesforce admin", "salesforce developer", "sales cloud", "service cloud"
    }),
    ("Low Code / No Code", {
        "low code", "no code", "low-code", "no-code", "appgyver", "outsystems", "mendix",
        "low code platform", "citizen developer", "bubble", "zapier"
    }),
    ("Database", {
        "database administrator", "dba", "database engineer", "database developer",
        "mysql", "postgresql", "oracle db", "oracle database", "sql server", 
        "mongodb", "redis", "cassandra", "database design", "nosql",
        "pl/sql", "plsql", "t-sql", "tsql", "sql developer", "database architect"
    }),
    ("Integration / APIs", {
        "integration", "api", "apis", "rest api", "restful api", "soap", "graphql",
        "microservices", "mule", "mulesoft", "boomi", "api integration", "enterprise integration",
        "tibco", "webmethods"
    }),
    ("UI/UX", {
        "ui designer", "ux designer", "ui/ux", "user interface", "user experience",
        "figma", "adobe xd", "sketch", "wireframing", "prototyping", "ui design", "ux design",
        "interaction design", "invision", "zeplin"
    }),
    ("Support", {
        "technical support", "it support", "support engineer", "help desk", "desktop support",
        "application support", "production support", "l1 support", "l2 support", "l3 support",
        "customer support", "support analyst", "troubleshooting", "incident management"
    }),
    ("Business Development", {
        "business development", "bd", "business dev", "bde", "business development executive",
        "business development manager", "b2b sales", "client acquisition", "market expansion",
        "partnership development", "strategic partnerships", "account development",
        "lead generation", "lead generation specialist", "sales development", "sales executive",
        "marketing executive", "email marketing", "e-mail marketing", "market research",
        "identifying prospects", "cold calling", "prospecting", "mba marketing"
    }),
    # New Profile Types (50 additional)
    ("Go / Golang", {
        "go", "golang", "go language", "go programming", "go developer",
        "goroutine", "go routines", "go modules", "go kit", "gin", "echo", "fiber"
    }),
    ("Ruby", {
        "ruby", "ruby on rails", "rails", "ruby developer", "ruby programmer",
        "sinatra", "rack", "rspec", "capybara", "rubygems", "bundler", "rake"
    }),
    ("PHP", {
        "php", "php developer", "php programmer", "laravel", "symfony", "codeigniter",
        "zend", "yii", "cakephp", "wordpress", "drupal", "magento", "composer"
    }),
    ("Rust", {
        "rust", "rust programming", "rust developer", "rustlang", "cargo", "rustc",
        "tokio", "actix", "rocket", "rust web", "rust backend"
    }),
    ("Scala", {
        "scala", "scala developer", "scala programming", "akka", "play framework",
        "spark scala", "scalatra", "sbt", "scala functional", "cats", "zio"
    }),
    ("C/C++", {
        "c++", "cpp", "c programming", "c++ programming", "c developer", "cpp developer",
        "qt", "boost", "stl", "cmake", "gcc", "clang", "visual c++", "mfc"
    }),
    ("React", {
        "react", "react.js", "reactjs", "react developer", "react native", "redux",
        "mobx", "next.js", "gatsby", "react hooks", "jsx", "react router"
    }),
    ("Angular", {
        "angular", "angularjs", "angular 2", "angular developer", "typescript angular",
        "angular cli", "rxjs", "ngrx", "angular material", "ionic angular"
    }),
    ("Vue.js", {
        "vue", "vue.js", "vuejs", "vue developer", "nuxt.js", "vuex", "vue router",
        "vuetify", "quasar", "vue 3", "composition api"
    }),
    ("Node.js", {
        "node.js", "nodejs", "node", "node developer", "express.js", "nest.js",
        "koa", "hapi", "socket.io", "npm", "yarn", "pm2", "node backend"
    }),
    ("Microservices", {
        "microservices", "microservice", "microservice architecture", "service mesh",
        "istio", "consul", "eureka", "api gateway", "distributed systems"
    }),
    ("Serverless", {
        "serverless", "serverless architecture", "aws lambda", "azure functions",
        "google cloud functions", "serverless framework", "faas", "function as a service"
    }),
    ("AWS", {
        "aws", "amazon web services", "aws cloud", "ec2", "s3", "lambda", "rds",
        "dynamodb", "cloudformation", "eks", "ecs", "iam", "vpc", "route53"
    }),
    ("Azure", {
        "azure", "microsoft azure", "azure cloud", "azure functions", "azure devops",
        "aks", "app service", "azure sql", "cosmos db", "azure ad", "arm templates"
    }),
    ("GCP", {
        "gcp", "google cloud", "google cloud platform", "gke", "cloud functions",
        "bigquery", "cloud storage", "app engine", "cloud run", "cloud sql"
    }),
    ("Kubernetes", {
        "kubernetes", "k8s", "kubernetes engineer", "kubectl", "helm", "kustomize",
        "kubernetes operator", "crd", "pod", "deployment", "service mesh"
    }),
    ("Docker", {
        "docker", "docker container", "dockerfile", "docker compose", "docker swarm",
        "containerization", "containers", "docker engine", "docker hub"
    }),
    ("Terraform", {
        "terraform", "terraform iac", "infrastructure as code", "terraform cloud",
        "terraform state", "terraform modules", "hcl", "terraform provider"
    }),
    ("Ansible", {
        "ansible", "ansible playbook", "ansible tower", "ansible automation",
        "ansible roles", "ansible vault", "configuration management"
    }),
    ("Jenkins", {
        "jenkins", "jenkins pipeline", "jenkinsfile", "jenkins ci/cd", "jenkins x",
        "blue ocean", "jenkins plugins", "continuous integration"
    }),
    ("GitLab CI/CD", {
        "gitlab ci", "gitlab cicd", "gitlab pipeline", "gitlab runner", ".gitlab-ci.yml",
        "gitlab devops", "gitlab automation"
    }),
    ("GitHub Actions", {
        "github actions", "github ci/cd", "github workflow", "github automation",
        "actions runner", "github pipelines"
    }),
    ("MongoDB", {
        "mongodb", "mongo", "mongodb developer", "mongoose", "mongodb atlas",
        "nosql mongodb", "document database", "mongodb compass"
    }),
    ("PostgreSQL", {
        "postgresql", "postgres", "postgresql developer", "postgresql dba",
        "pgadmin", "postgis", "postgresql performance", "plpgsql"
    }),
    ("MySQL", {
        "mysql", "mysql developer", "mysql dba", "mysql database", "mariadb",
        "mysql performance", "mysql optimization", "innodb", "myisam"
    }),
    ("Redis", {
        "redis", "redis cache", "redis developer", "redis cluster", "redis sentinel",
        "redis pub/sub", "redis streams", "redis cache"
    }),
    ("Elasticsearch", {
        "elasticsearch", "elastic", "elasticsearch developer", "elk stack", "kibana",
        "logstash", "elastic apm", "elastic cloud", "search engine"
    }),
    ("Apache Kafka", {
        "kafka", "apache kafka", "kafka developer", "kafka streams", "kafka connect",
        "kafka producer", "kafka consumer", "event streaming", "confluent"
    }),
    ("Apache Spark", {
        "spark", "apache spark", "spark developer", "pyspark", "spark sql",
        "spark streaming", "databricks spark", "spark ml", "spark rdd"
    }),
    ("Hadoop", {
        "hadoop", "apache hadoop", "hadoop developer", "hdfs", "mapreduce",
        "yarn", "hive", "pig", "hbase", "hadoop ecosystem"
    }),
    ("Machine Learning", {
        "machine learning", "ml", "ml engineer", "ml developer", "scikit-learn",
        "xgboost", "lightgbm", "mlops", "model training", "feature engineering"
    }),
    ("Deep Learning", {
        "deep learning", "neural networks", "cnn", "rnn", "lstm", "transformer",
        "pytorch", "tensorflow", "keras", "deep neural networks", "ai models"
    }),
    ("Computer Vision", {
        "computer vision", "cv", "opencv", "image processing", "image recognition",
        "object detection", "yolo", "faster r-cnn", "image classification"
    }),
    ("NLP", {
        "nlp", "natural language processing", "nlp engineer", "text processing",
        "sentiment analysis", "named entity recognition", "bert", "gpt", "transformer"
    }),
    ("Blockchain", {
        "blockchain", "blockchain developer", "blockchain engineer", "ethereum",
        "solidity", "smart contracts", "web3", "defi", "cryptocurrency"
    }),
    ("Web3", {
        "web3", "web 3", "web3 developer", "ethereum", "solidity", "smart contracts",
        "nft", "defi", "dapp", "metamask", "ipfs", "polygon"
    }),
    ("IoT", {
        "iot", "internet of things", "iot developer", "iot engineer", "embedded iot",
        "arduino", "raspberry pi", "mqtt", "iot sensors", "edge computing"
    }),
    ("Embedded Systems", {
        "embedded systems", "embedded developer", "embedded engineer", "firmware",
        "microcontroller", "arm", "stm32", "embedded c", "rtos", "bare metal"
    }),
    ("Game Development", {
        "game development", "game developer", "unity", "unreal engine", "game engine",
        "c# unity", "c++ game", "game programming", "game design", "gamedev"
    }),
    ("AR/VR", {
        "ar", "vr", "augmented reality", "virtual reality", "ar developer", "vr developer",
        "unity ar", "unreal vr", "oculus", "hololens", "ar/vr", "mixed reality"
    }),
    ("FinTech", {
        "fintech", "financial technology", "fintech developer", "payment systems",
        "banking software", "trading systems", "cryptocurrency", "blockchain finance"
    }),
    ("Healthcare IT", {
        "healthcare it", "healthcare software", "hl7", "fhir", "ehr", "emr",
        "health informatics", "medical software", "healthcare systems"
    }),
    ("E-commerce", {
        "ecommerce", "e-commerce", "ecommerce developer", "online shopping",
        "payment gateway", "shopping cart", "magento", "shopify", "woocommerce"
    }),
    ("Content Management", {
        "cms", "content management", "wordpress", "drupal", "joomla", "contentful",
        "headless cms", "strapi", "ghost", "cms developer"
    }),
    ("Video Streaming", {
        "video streaming", "streaming media", "ffmpeg", "video processing", "hls",
        "dash", "webrtc", "video codec", "streaming platform", "video engineer"
    }),
    ("Network Engineering", {
        "network engineer", "network administrator", "ccna", "ccnp", "cisco",
        "routing", "switching", "firewall", "vpn", "network security", "tcp/ip"
    }),
    ("System Administration", {
        "system administrator", "sysadmin", "linux admin", "windows admin", "unix",
        "server administration", "system management", "it operations", "infrastructure"
    }),
    ("GraphQL", {
        "graphql", "graph ql", "graphql api", "graphql developer", "apollo",
        "relay", "graphql schema", "graphql query", "graphql mutation"
    }),
    ("TypeScript", {
        "typescript", "ts", "typescript developer", "tsx", "typescript programming",
        "angular typescript", "react typescript", "node typescript"
    }),
    ("Linux", {
        "linux", "linux admin", "linux developer", "linux system", "ubuntu", "centos",
        "red hat", "debian", "bash scripting", "shell scripting", "linux kernel"
    }),
    # Additional 50 Profile Types
    ("Swift", {
        "swift", "swift programming", "swift developer", "ios swift", "swiftui",
        "swift language", "apple swift", "swift ios", "swift macos"
    }),
    ("Kotlin", {
        "kotlin", "kotlin developer", "kotlin android", "kotlin programming",
        "kotlin coroutines", "kotlin multiplatform", "android kotlin"
    }),
    ("Perl", {
        "perl", "perl programming", "perl developer", "perl scripting", "cpan"
    }),
    ("Shell Scripting", {
        "shell scripting", "bash", "shell script", "bash scripting", "zsh",
        "shell programming", "bash developer", "unix shell"
    }),
    ("PowerShell", {
        "powershell", "powershell scripting", "powershell automation", "ps1",
        "powershell developer", "azure powershell"
    }),
    ("Groovy", {
        "groovy", "groovy programming", "groovy developer", "apache groovy",
        "gradle groovy", "groovy scripting"
    }),
    ("Clojure", {
        "clojure", "clojure programming", "clojure developer", "clojurescript"
    }),
    ("Erlang", {
        "erlang", "erlang programming", "erlang developer", "elixir erlang"
    }),
    ("Elixir", {
        "elixir", "elixir programming", "elixir developer", "phoenix framework",
        "elixir phoenix", "elixir otp"
    }),
    ("Haskell", {
        "haskell", "haskell programming", "haskell developer", "functional haskell"
    }),
    ("F#", {
        "f#", "fsharp", "f sharp", "f# programming", "f# developer", ".net f#"
    }),
    ("VB.NET", {
        "vb.net", "vbnet", "visual basic", "vb.net programming", "vb.net developer"
    }),
    ("COBOL", {
        "cobol", "cobol programming", "cobol developer", "mainframe cobol"
    }),
    ("Fortran", {
        "fortran", "fortran programming", "fortran developer", "scientific computing", "scientific fortran"
    }),
    ("Assembly", {
        "assembly", "assembly language", "asm", "x86 assembly", "arm assembly"
    }),
    ("MATLAB", {
        "matlab", "matlab programming", "matlab developer", "matlab simulink",
        "mathematical computing", "matlab scripting"
    }),
    ("R", {
        "r programming", "r language", "r developer", "r statistical", "rstudio",
        "r data analysis", "r programming language"
    }),
    ("Julia", {
        "julia", "julia programming", "julia developer", "julia language",
        "scientific julia"
    }),
    ("Lua", {
        "lua", "lua programming", "lua developer", "lua scripting", "lua game"
    }),
    ("Dart", {
        "dart", "dart programming", "dart developer", "flutter dart", "dart language"
    }),
    ("Objective-C", {
        "objective-c", "objective c", "objc", "objective-c developer", "ios objective-c"
    }),
    ("Delphi", {
        "delphi", "delphi programming", "delphi developer", "pascal delphi"
    }),
    ("Pascal", {
        "pascal", "pascal programming", "pascal developer", "object pascal"
    }),
    ("Ada", {
        "ada", "ada programming", "ada developer", "ada language"
    }),
    ("Prolog", {
        "prolog", "prolog programming", "prolog developer", "logic programming"
    }),
    ("Lisp", {
        "lisp", "lisp programming", "common lisp", "scheme", "clojure lisp"
    }),
    ("Smalltalk", {
        "smalltalk", "smalltalk programming", "smalltalk developer"
    }),
    ("OCaml", {
        "ocaml", "ocaml programming", "ocaml developer", "functional ocaml"
    }),
    ("Racket", {
        "racket", "racket programming", "racket developer", "racket language"
    }),
    ("Crystal", {
        "crystal", "crystal programming", "crystal developer", "crystal language"
    }),
    ("Nim", {
        "nim", "nim programming", "nim developer", "nim language"
    }),
    ("Zig", {
        "zig", "zig programming", "zig developer", "zig language"
    }),
    ("V", {
        "v language", "v programming", "v developer", "vlang"
    }),
    ("D", {
        "d programming", "d language", "d developer", "dlang"
    }),
    ("Nix", {
        "nix", "nixos", "nix package manager", "nix developer"
    }),
    ("Terraform Cloud", {
        "terraform cloud", "terraform enterprise", "terraform sentinel"
    }),
    ("Pulumi", {
        "pulumi", "pulumi iac", "pulumi developer", "infrastructure pulumi"
    }),
    ("CloudFormation", {
        "cloudformation", "aws cloudformation", "cfn", "cloudformation templates"
    }),
    ("ARM Templates", {
        "arm templates", "azure resource manager", "arm bicep", "azure arm"
    }),
    ("Bicep", {
        "bicep", "azure bicep", "bicep language", "bicep iac"
    }),
    ("CDK", {
        "cdk", "aws cdk", "cloud development kit", "cdk typescript", "cdk python"
    }),
    ("Serverless Framework", {
        "serverless framework", "serverless.yml", "serverless deploy", "serverless plugin"
    }),
    ("SAM", {
        "sam", "aws sam", "serverless application model", "sam template"
    }),
    ("Zappa", {
        "zappa", "zappa python", "zappa serverless", "python zappa"
    }),
    ("Chalice", {
        "chalice", "aws chalice", "python chalice", "serverless chalice"
    }),
    ("Vercel", {
        "vercel", "vercel deploy", "vercel platform", "next.js vercel"
    }),
    ("Netlify", {
        "netlify", "netlify deploy", "netlify functions", "netlify cms"
    }),
    ("Firebase", {
        "firebase", "firebase developer", "firebase functions", "firebase hosting",
        "firestore", "firebase auth", "firebase cloud"
    }),
    ("Supabase", {
        "supabase", "supabase developer", "supabase database", "postgres supabase"
    }),
    ("Appwrite", {
        "appwrite", "appwrite backend", "appwrite developer"
    }),
    ("Hasura", {
        "hasura", "hasura graphql", "hasura developer", "graphql hasura"
    }),
]

# Enhanced rules with weights and expanded keywords
PROFILE_TYPE_RULES_ENHANCED: List[Tuple[str, Dict[str, float]]] = [
    (
        "Java",
        {
            "java": 5.0, "jdk": 4.5, "jvm": 4.5, "jre": 4.0,
            "spring boot": 4.5, "spring": 4.0, "spring mvc": 3.5, "spring cloud": 3.5,
            "hibernate": 3.5, "jpa": 3.0, "j2ee": 3.0, "jakarta ee": 3.0,
            "servlet": 2.5, "jsp": 2.5, "jsf": 2.5, "struts": 2.0,
            "microservices": 2.0, "rest api": 2.0, "soap": 1.5,
            "maven": 2.0, "gradle": 2.0,
        }
    ),
    (
        ".Net",
        {
            "c#": 5.0, "csharp": 5.0, "c sharp": 5.0,
            "asp.net": 4.5, "asp.net core": 4.5, ".net core": 4.5,
            ".net framework": 4.0, "dotnet": 4.0, ".net": 3.5,
            "entity framework": 3.5, "ef core": 3.5, "mvc": 3.0,
            "wpf": 2.5, "winforms": 2.5, "web api": 2.5,
            "linq": 2.0, "xamarin": 2.0, "blazor": 2.0,
        }
    ),
    (
        "Python",
        {
            "python": 5.0, "django": 4.0, "flask": 3.5, "fastapi": 3.5,
            "tornado": 2.5, "pyramid": 2.0,
            "pandas": 3.0, "numpy": 2.5, "scipy": 2.0,
            "matplotlib": 2.0, "seaborn": 1.5, "sqlalchemy": 2.5,
            "celery": 2.0, "redis": 1.5, "asyncio": 1.5,
        }
    ),
    (
        "JavaScript",
        {
            "javascript": 5.0, "js": 4.5, "typescript": 4.5,
            "node.js": 4.5, "node": 4.0,
            "react": 4.0, "angular": 3.5, "vue": 3.5,
            "es6": 3.0, "jquery": 2.5, "express": 3.0,
            "next.js": 3.0, "nuxt.js": 2.5, "nuxt": 2.5,
            "webpack": 2.0, "babel": 1.5,
        }
    ),
    (
        "Full Stack",
        {
            "full stack": 5.0, "fullstack": 4.5,
            "mern": 4.0, "mean": 4.0, "mevn": 3.5, "lamp": 3.0,
            "next.js": 3.0, "nuxt": 2.5,
            "react": 2.0, "node.js": 2.0, "express": 2.0,
            "mongodb": 2.0, "postgresql": 1.5, "mysql": 1.5,
        }
    ),
    (
        "DevOps",
        {
            "devops": 5.0, "ci/cd": 4.0, "cicd": 4.0,
            "kubernetes": 4.5, "docker": 4.0, "helm": 3.0,
            "terraform": 3.5, "ansible": 3.0, "chef": 2.5, "puppet": 2.5,
            "jenkins": 3.0, "gitlab ci": 2.5, "github actions": 2.5,
            "monitoring": 2.0, "prometheus": 2.0, "grafana": 1.5,
        }
    ),
    (
        "Data Engineering",
        {
            "data engineer": 5.0, "data engineering": 4.5,
            "etl": 4.0, "elt": 3.5, "airflow": 3.5,
            "snowflake": 3.5, "spark": 3.5, "pyspark": 3.0,
            "hadoop": 3.0, "hdfs": 2.5, "databricks": 3.0,
            "kafka": 2.5, "flume": 2.0, "sqoop": 2.0,
            "redshift": 2.0, "bigquery": 2.0,
        }
    ),
    (
        "Data Science",
        {
            "data science": 5.0, "machine learning": 4.5,
            "deep learning": 3.5, "ml engineer": 4.0,
            "ai engineer": 4.0, "artificial intelligence": 3.5,
            "llm": 3.5, "nlp": 3.0, "computer vision": 2.5,
            "tensorflow": 2.5, "pytorch": 2.5, "scikit-learn": 2.0,
        }
    ),
    (
        "Testing / QA",
        {
            "qa": 4.0, "quality assurance": 4.5,
            "manual testing": 3.0, "automation testing": 4.0,
            "selenium": 3.5, "cypress": 3.0, "playwright": 2.5,
            "junit": 2.5, "testng": 2.5,
            "postman": 2.0, "api testing": 2.0,
            "performance testing": 2.0, "jmeter": 2.0,
        }
    ),
    (
        "SAP",
        {
            "sap": 5.0, "abap": 4.0,
            "sap hana": 3.5, "s/4hana": 3.5, "hana": 3.5,
            "fiori": 2.5, "sap mm": 2.5, "sap sd": 2.5,
            "sap fico": 2.5, "successfactors": 3.0,
            "ariba": 2.5, "sap basis": 2.0,
        }
    ),
    (
        "ERP",
        {
            "erp": 4.0,
            "oracle apps": 3.5, "oracle e-business": 3.5, "oracle fusion": 3.0,
            "d365": 3.0, "dynamics 365": 3.0, "dynamics": 3.0,
            "business central": 2.5, "netsuite": 2.5,
            "workday": 2.5, "odoo": 2.0, "peopleSoft": 2.0,
        }
    ),
    (
        "Cloud / Infra",
        {
            "aws": 4.0, "azure": 4.0, "gcp": 3.5, "google cloud": 3.5,
            "ec2": 2.5, "s3": 2.5, "lambda": 2.0,
            "iam": 2.0, "eks": 2.5, "aks": 2.5,
            "cloud architect": 4.5, "cloud engineer": 4.0,
            "terraform": 3.0, "cloudformation": 2.0, "vmware": 2.0,
        }
    ),
    (
        "Business Intelligence (BI)",
        {
            "power bi": 5.0, "tableau": 4.5, "qlik": 4.0, "looker": 3.5,
            "ssis": 3.0, "ssrs": 3.0, "business intelligence": 4.0,
            "bi": 3.5, "data visualization": 2.5, "dashboard": 2.0,
        }
    ),
    (
        "Microsoft Power Platform",
        {
            "power platform": 5.0, "power apps": 4.5, "power automate": 4.0,
            "power bi": 4.0, "power virtual agents": 3.5,
            "dataverse": 3.5, "power fx": 3.0, "canvas app": 2.5,
            "model-driven app": 2.5, "power pages": 2.0,
        }
    ),
    (
        "RPA",
        {
            "rpa": 5.0, "robotic process automation": 4.5,
            "ui path": 4.5, "uipath": 4.5,
            "automation anywhere": 4.0, "blue prism": 3.5,
            "rpa developer": 4.0, "process automation": 3.0,
        }
    ),
    (
        "Cyber Security",
        {
            "cyber security": 5.0, "cybersecurity": 5.0, "information security": 4.5,
            "soc": 4.0, "security operations center": 4.0,
            "siem": 3.5, "security information": 3.5,
            "penetration testing": 3.5, "pen testing": 3.5,
            "ethical hacking": 3.0, "vapt": 3.0, "vulnerability assessment": 3.0,
            "security analyst": 3.5, "security engineer": 3.5,
        }
    ),
    (
        "Mobile Development",
        {
            "android": 4.5, "ios": 4.5, "mobile development": 5.0,
            "kotlin": 4.0, "swift": 4.0,
            "flutter": 3.5, "react native": 3.5, "xamarin": 3.0,
            "mobile app": 3.0, "ios development": 3.5, "android development": 3.5,
        }
    ),
    (
        "Salesforce",
        {
            "salesforce": 5.0, "sfdc": 4.0,
            "apex": 4.0, "visualforce": 3.5,
            "lightning": 3.5, "lightning web components": 3.5, "lwc": 3.5,
            "soql": 3.0, "sosl": 3.0, "salesforce admin": 3.0,
            "salesforce developer": 4.0, "sales cloud": 2.5, "service cloud": 2.5,
        }
    ),
    (
        "Low Code / No Code",
        {
            "low code": 5.0, "no code": 4.5, "low-code": 5.0, "no-code": 4.5,
            "appgyver": 3.5, "outsystems": 3.5, "mendix": 3.5,
            "low code platform": 4.0, "citizen developer": 2.5,
        }
    ),
    (
        "Database",
        {
            # High-confidence technical database terms
            "database administrator": 5.0, "dba": 5.0, "database engineer": 5.0,
            "database developer": 4.5, "database design": 4.0, "database architect": 4.5,
            "mysql": 4.0, "postgresql": 4.0, "oracle db": 4.0, "oracle database": 4.0,
            "sql server": 4.0, "mongodb": 3.5, "redis": 3.0, "cassandra": 3.0,
            "pl/sql": 4.0, "plsql": 4.0, "t-sql": 4.0, "tsql": 4.0,
            "sql query": 3.5, "sql developer": 4.0, "nosql": 3.0,
            "database management": 3.5, "database optimization": 4.0,
            "database tuning": 4.0, "query optimization": 3.5,
            # Low score for standalone "database" to avoid false positives
            # (e.g., "lead database", "prospect database" in sales context)
            "database": 1.5, "sql": 2.0,
        }
    ),
    (
        "Integration / APIs",
        {
            "integration": 4.0, "api": 3.5, "apis": 3.5,
            "rest api": 3.5, "restful api": 3.5, "soap": 3.0,
            "graphql": 3.0, "microservices": 3.0,
            "mule": 3.5, "mulesoft": 3.5, "boomi": 3.0,
            "api integration": 3.5, "enterprise integration": 3.0,
        }
    ),
    (
        "UI/UX",
        {
            "ui designer": 4.5, "ux designer": 4.5, "ui/ux": 5.0,
            "user interface": 4.0, "user experience": 4.0,
            "figma": 3.5, "adobe xd": 3.0, "sketch": 3.0,
            "wireframing": 3.0, "prototyping": 2.5, "ui design": 3.5,
            "ux design": 3.5, "interaction design": 2.5,
        }
    ),
    (
        "Support",
        {
            "technical support": 5.0, "it support": 4.5, "support engineer": 4.5,
            "help desk": 4.0, "desktop support": 3.5,
            "application support": 4.0, "production support": 4.0,
            "l1 support": 3.0, "l2 support": 3.5, "l3 support": 4.0,
            "customer support": 3.0, "support analyst": 3.5,
        }
    ),
    (
        "Business Development",
        {
            "business development": 5.0, "bd": 4.5, "business dev": 4.5,
            "bde": 4.0, "business development executive": 4.5,
            "business development manager": 4.5, "b2b sales": 4.0,
            "client acquisition": 3.5, "market expansion": 3.5,
            "partnership development": 3.5, "strategic partnerships": 3.5,
            "account development": 3.0,
            # Sales & Marketing keywords
            "lead generation": 5.0, "lead generation specialist": 5.0,
            "sales development": 4.5, "sales development executive": 4.5,
            "sales executive": 4.0, "marketing executive": 4.0,
            "email marketing": 4.0, "e-mail marketing": 4.0,
            "market research": 3.5, "identifying prospects": 4.0,
            "cold calling": 3.5, "prospecting": 3.5,
            "crm": 3.0, "salesforce crm": 3.0, "hubspot": 3.0,
            "linkedin sales": 3.5, "zoominfo": 3.5,
            "mba marketing": 4.0, "sales and marketing": 4.5,
        }
    ),
    # New Profile Types Enhanced (50 additional)
    (
        "Go / Golang",
        {
            "go": 5.0, "golang": 5.0, "go language": 4.5, "go programming": 4.5,
            "go developer": 4.5, "goroutine": 3.5, "go routines": 3.5,
            "go modules": 3.0, "go kit": 2.5, "gin": 3.0, "echo": 2.5, "fiber": 2.5,
        }
    ),
    (
        "Ruby",
        {
            "ruby": 5.0, "ruby on rails": 4.5, "rails": 4.5, "ruby developer": 4.5,
            "ruby programmer": 4.0, "sinatra": 3.0, "rack": 2.5, "rspec": 2.5,
            "capybara": 2.0, "rubygems": 2.0, "bundler": 2.0, "rake": 2.0,
        }
    ),
    (
        "PHP",
        {
            "php": 5.0, "php developer": 4.5, "php programmer": 4.0,
            "laravel": 4.0, "symfony": 3.5, "codeigniter": 3.0, "zend": 2.5,
            "yii": 2.5, "cakephp": 2.5, "wordpress": 3.0, "drupal": 2.5,
            "magento": 3.0, "composer": 2.5,
        }
    ),
    (
        "Rust",
        {
            "rust": 5.0, "rust programming": 4.5, "rust developer": 4.5,
            "rustlang": 4.0, "cargo": 3.5, "rustc": 3.0, "tokio": 3.0,
            "actix": 2.5, "rocket": 2.5, "rust web": 2.5, "rust backend": 2.5,
        }
    ),
    (
        "Scala",
        {
            "scala": 5.0, "scala developer": 4.5, "scala programming": 4.5,
            "akka": 3.5, "play framework": 3.5, "spark scala": 3.0,
            "scalatra": 2.5, "sbt": 2.5, "scala functional": 3.0, "cats": 2.0, "zio": 2.0,
        }
    ),
    (
        "C/C++",
        {
            "c++": 5.0, "cpp": 4.5, "c programming": 4.0, "c++ programming": 4.5,
            "c developer": 4.0, "cpp developer": 4.5, "qt": 3.5, "boost": 3.0,
            "stl": 3.0, "cmake": 2.5, "gcc": 2.5, "clang": 2.5, "visual c++": 3.0, "mfc": 2.5,
        }
    ),
    (
        "React",
        {
            "react": 5.0, "react.js": 4.5, "reactjs": 4.5, "react developer": 4.5,
            "react native": 3.5, "redux": 3.5, "mobx": 2.5, "next.js": 3.0,
            "gatsby": 2.5, "react hooks": 3.0, "jsx": 2.5, "react router": 2.5,
        }
    ),
    (
        "Angular",
        {
            "angular": 5.0, "angularjs": 4.0, "angular 2": 4.5, "angular developer": 4.5,
            "typescript angular": 3.5, "angular cli": 3.0, "rxjs": 3.0, "ngrx": 2.5,
            "angular material": 2.5, "ionic angular": 2.5,
        }
    ),
    (
        "Vue.js",
        {
            "vue": 5.0, "vue.js": 4.5, "vuejs": 4.5, "vue developer": 4.5,
            "nuxt.js": 3.5, "vuex": 3.0, "vue router": 2.5, "vuetify": 2.5,
            "quasar": 2.0, "vue 3": 3.0, "composition api": 2.5,
        }
    ),
    (
        "Node.js",
        {
            "node.js": 5.0, "nodejs": 4.5, "node": 4.0, "node developer": 4.5,
            "express.js": 3.5, "nest.js": 3.0, "koa": 2.5, "hapi": 2.0,
            "socket.io": 2.5, "npm": 2.0, "yarn": 2.0, "pm2": 2.0, "node backend": 3.0,
        }
    ),
    (
        "Microservices",
        {
            "microservices": 5.0, "microservice": 4.5, "microservice architecture": 4.5,
            "service mesh": 3.5, "istio": 3.0, "consul": 2.5, "eureka": 2.5,
            "api gateway": 3.0, "distributed systems": 3.0,
        }
    ),
    (
        "Serverless",
        {
            "serverless": 5.0, "serverless architecture": 4.5, "aws lambda": 3.5,
            "azure functions": 3.0, "google cloud functions": 3.0, "serverless framework": 3.5,
            "faas": 3.0, "function as a service": 3.0,
        }
    ),
    (
        "AWS",
        {
            "aws": 5.0, "amazon web services": 4.5, "aws cloud": 4.0, "ec2": 3.5,
            "s3": 3.5, "lambda": 3.0, "rds": 3.0, "dynamodb": 3.0, "cloudformation": 2.5,
            "eks": 3.0, "ecs": 2.5, "iam": 2.5, "vpc": 2.5, "route53": 2.0,
        }
    ),
    (
        "Azure",
        {
            "azure": 5.0, "microsoft azure": 4.5, "azure cloud": 4.0, "azure functions": 3.5,
            "azure devops": 3.5, "aks": 3.0, "app service": 3.0, "azure sql": 3.0,
            "cosmos db": 2.5, "azure ad": 2.5, "arm templates": 2.5,
        }
    ),
    (
        "GCP",
        {
            "gcp": 5.0, "google cloud": 4.5, "google cloud platform": 4.5, "gke": 3.5,
            "cloud functions": 3.0, "bigquery": 3.0, "cloud storage": 2.5,
            "app engine": 2.5, "cloud run": 2.5, "cloud sql": 2.5,
        }
    ),
    (
        "Kubernetes",
        {
            "kubernetes": 5.0, "k8s": 4.5, "kubernetes engineer": 4.5, "kubectl": 3.5,
            "helm": 3.5, "kustomize": 2.5, "kubernetes operator": 3.0, "crd": 2.5,
            "pod": 2.0, "deployment": 2.0, "service mesh": 2.5,
        }
    ),
    (
        "Docker",
        {
            "docker": 5.0, "docker container": 4.5, "dockerfile": 4.0, "docker compose": 3.5,
            "docker swarm": 3.0, "containerization": 3.5, "containers": 3.0,
            "docker engine": 2.5, "docker hub": 2.0,
        }
    ),
    (
        "Terraform",
        {
            "terraform": 5.0, "terraform iac": 4.5, "infrastructure as code": 4.0,
            "terraform cloud": 3.0, "terraform state": 3.0, "terraform modules": 3.0,
            "hcl": 2.5, "terraform provider": 2.5,
        }
    ),
    (
        "Ansible",
        {
            "ansible": 5.0, "ansible playbook": 4.0, "ansible tower": 3.5,
            "ansible automation": 3.5, "ansible roles": 3.0, "ansible vault": 2.5,
            "configuration management": 3.0,
        }
    ),
    (
        "Jenkins",
        {
            "jenkins": 5.0, "jenkins pipeline": 4.0, "jenkinsfile": 4.0, "jenkins ci/cd": 4.0,
            "jenkins x": 3.0, "blue ocean": 2.5, "jenkins plugins": 2.5,
            "continuous integration": 3.0,
        }
    ),
    (
        "GitLab CI/CD",
        {
            "gitlab ci": 5.0, "gitlab cicd": 4.5, "gitlab pipeline": 4.0, "gitlab runner": 3.5,
            ".gitlab-ci.yml": 3.5, "gitlab devops": 3.0, "gitlab automation": 3.0,
        }
    ),
    (
        "GitHub Actions",
        {
            "github actions": 5.0, "github ci/cd": 4.5, "github workflow": 4.0,
            "github automation": 3.5, "actions runner": 3.0, "github pipelines": 3.0,
        }
    ),
    (
        "MongoDB",
        {
            "mongodb": 5.0, "mongo": 4.5, "mongodb developer": 4.5, "mongoose": 3.5,
            "mongodb atlas": 3.0, "nosql mongodb": 3.0, "document database": 2.5,
            "mongodb compass": 2.0,
        }
    ),
    (
        "PostgreSQL",
        {
            "postgresql": 5.0, "postgres": 4.5, "postgresql developer": 4.5, "postgresql dba": 4.0,
            "pgadmin": 2.5, "postgis": 2.5, "postgresql performance": 3.0, "plpgsql": 3.0,
        }
    ),
    (
        "MySQL",
        {
            "mysql": 5.0, "mysql developer": 4.5, "mysql dba": 4.0, "mysql database": 3.5,
            "mariadb": 3.0, "mysql performance": 3.0, "mysql optimization": 3.0,
            "innodb": 2.5, "myisam": 2.0,
        }
    ),
    (
        "Redis",
        {
            "redis": 5.0, "redis cache": 4.0, "redis developer": 4.5, "redis cluster": 3.5,
            "redis sentinel": 3.0, "redis pub/sub": 2.5, "redis streams": 2.5, "redis cache": 3.5,
        }
    ),
    (
        "Elasticsearch",
        {
            "elasticsearch": 5.0, "elastic": 4.0, "elasticsearch developer": 4.5, "elk stack": 4.0,
            "kibana": 3.5, "logstash": 3.0, "elastic apm": 2.5, "elastic cloud": 2.5,
            "search engine": 2.5,
        }
    ),
    (
        "Apache Kafka",
        {
            "kafka": 5.0, "apache kafka": 4.5, "kafka developer": 4.5, "kafka streams": 3.5,
            "kafka connect": 3.0, "kafka producer": 2.5, "kafka consumer": 2.5,
            "event streaming": 3.0, "confluent": 2.5,
        }
    ),
    (
        "Apache Spark",
        {
            "spark": 5.0, "apache spark": 4.5, "spark developer": 4.5, "pyspark": 3.5,
            "spark sql": 3.5, "spark streaming": 3.0, "databricks spark": 3.0,
            "spark ml": 2.5, "spark rdd": 2.5,
        }
    ),
    (
        "Hadoop",
        {
            "hadoop": 5.0, "apache hadoop": 4.5, "hadoop developer": 4.5, "hdfs": 4.0,
            "mapreduce": 3.5, "yarn": 3.0, "hive": 3.0, "pig": 2.5, "hbase": 2.5,
            "hadoop ecosystem": 3.0,
        }
    ),
    (
        "Machine Learning",
        {
            "machine learning": 5.0, "ml": 4.0, "ml engineer": 4.5, "ml developer": 4.0,
            "scikit-learn": 3.5, "xgboost": 3.0, "lightgbm": 2.5, "mlops": 3.0,
            "model training": 3.0, "feature engineering": 2.5,
        }
    ),
    (
        "Deep Learning",
        {
            "deep learning": 5.0, "neural networks": 4.0, "cnn": 3.0, "rnn": 2.5,
            "lstm": 2.5, "transformer": 3.0, "pytorch": 3.5, "tensorflow": 3.5,
            "keras": 3.0, "deep neural networks": 3.5, "ai models": 2.5,
        }
    ),
    (
        "Computer Vision",
        {
            "computer vision": 5.0, "cv": 4.0, "opencv": 4.0, "image processing": 3.5,
            "image recognition": 3.5, "object detection": 3.0, "yolo": 2.5,
            "faster r-cnn": 2.0, "image classification": 3.0,
        }
    ),
    (
        "NLP",
        {
            "nlp": 5.0, "natural language processing": 4.5, "nlp engineer": 4.5,
            "text processing": 3.0, "sentiment analysis": 3.0, "named entity recognition": 2.5,
            "bert": 3.0, "gpt": 2.5, "transformer": 2.5,
        }
    ),
    (
        "Blockchain",
        {
            "blockchain": 5.0, "blockchain developer": 4.5, "blockchain engineer": 4.5,
            "ethereum": 4.0, "solidity": 4.0, "smart contracts": 4.0, "web3": 3.0,
            "defi": 2.5, "cryptocurrency": 2.5,
        }
    ),
    (
        "Web3",
        {
            "web3": 5.0, "web 3": 4.5, "web3 developer": 4.5, "ethereum": 3.5,
            "solidity": 3.5, "smart contracts": 3.5, "nft": 2.5, "defi": 2.5,
            "dapp": 3.0, "metamask": 2.0, "ipfs": 2.0, "polygon": 2.0,
        }
    ),
    (
        "IoT",
        {
            "iot": 5.0, "internet of things": 4.5, "iot developer": 4.5, "iot engineer": 4.5,
            "embedded iot": 3.5, "arduino": 3.0, "raspberry pi": 3.0, "mqtt": 3.0,
            "iot sensors": 2.5, "edge computing": 2.5,
        }
    ),
    (
        "Embedded Systems",
        {
            "embedded systems": 5.0, "embedded developer": 4.5, "embedded engineer": 4.5,
            "firmware": 4.0, "microcontroller": 3.5, "arm": 3.0, "stm32": 2.5,
            "embedded c": 3.5, "rtos": 3.0, "bare metal": 2.5,
        }
    ),
    (
        "Game Development",
        {
            "game development": 5.0, "game developer": 4.5, "unity": 4.0, "unreal engine": 4.0,
            "game engine": 3.5, "c# unity": 3.5, "c++ game": 3.0, "game programming": 3.5,
            "game design": 2.5, "gamedev": 3.0,
        }
    ),
    (
        "AR/VR",
        {
            "ar": 4.0, "vr": 4.0, "augmented reality": 4.5, "virtual reality": 4.5,
            "ar developer": 4.0, "vr developer": 4.0, "unity ar": 3.0, "unreal vr": 3.0,
            "oculus": 2.5, "hololens": 2.5, "ar/vr": 4.0, "mixed reality": 3.5,
        }
    ),
    (
        "FinTech",
        {
            "fintech": 5.0, "financial technology": 4.5, "fintech developer": 4.5,
            "payment systems": 3.5, "banking software": 3.5, "trading systems": 3.0,
            "cryptocurrency": 2.5, "blockchain finance": 2.5,
        }
    ),
    (
        "Healthcare IT",
        {
            "healthcare it": 5.0, "healthcare software": 4.5, "hl7": 3.5, "fhir": 3.5,
            "ehr": 3.5, "emr": 3.5, "health informatics": 3.0, "medical software": 3.5,
            "healthcare systems": 3.0,
        }
    ),
    (
        "E-commerce",
        {
            "ecommerce": 5.0, "e-commerce": 4.5, "ecommerce developer": 4.5,
            "online shopping": 3.0, "payment gateway": 3.0, "shopping cart": 2.5,
            "magento": 3.0, "shopify": 2.5, "woocommerce": 2.5,
        }
    ),
    (
        "Content Management",
        {
            "cms": 4.0, "content management": 4.5, "wordpress": 3.5, "drupal": 3.0,
            "joomla": 2.5, "contentful": 2.5, "headless cms": 3.0, "strapi": 2.5,
            "ghost": 2.0, "cms developer": 3.5,
        }
    ),
    (
        "Video Streaming",
        {
            "video streaming": 5.0, "streaming media": 4.0, "ffmpeg": 3.5, "video processing": 3.5,
            "hls": 3.0, "dash": 2.5, "webrtc": 3.0, "video codec": 2.5,
            "streaming platform": 3.0, "video engineer": 4.0,
        }
    ),
    (
        "Network Engineering",
        {
            "network engineer": 5.0, "network administrator": 4.5, "ccna": 3.5, "ccnp": 3.5,
            "cisco": 3.5, "routing": 3.0, "switching": 3.0, "firewall": 3.0,
            "vpn": 2.5, "network security": 3.0, "tcp/ip": 2.5,
        }
    ),
    (
        "System Administration",
        {
            "system administrator": 5.0, "sysadmin": 4.5, "linux admin": 4.0, "windows admin": 3.5,
            "unix": 3.0, "server administration": 3.5, "system management": 3.0,
            "it operations": 3.0, "infrastructure": 2.5,
        }
    ),
    (
        "GraphQL",
        {
            "graphql": 5.0, "graph ql": 4.5, "graphql api": 4.0, "graphql developer": 4.5,
            "apollo": 3.5, "relay": 2.5, "graphql schema": 3.0, "graphql query": 2.5,
            "graphql mutation": 2.5,
        }
    ),
    (
        "TypeScript",
        {
            "typescript": 5.0, "ts": 4.0, "typescript developer": 4.5, "tsx": 3.0,
            "typescript programming": 4.0, "angular typescript": 3.0, "react typescript": 3.0,
            "node typescript": 3.0,
        }
    ),
    (
        "Linux",
        {
            "linux": 5.0, "linux admin": 4.5, "linux developer": 4.0, "linux system": 3.5,
            "ubuntu": 3.0, "centos": 2.5, "red hat": 2.5, "debian": 2.5,
            "bash scripting": 3.0, "shell scripting": 3.0, "linux kernel": 2.5,
        }
    ),
    # Additional 50 Profile Types Enhanced
    (
        "Swift",
        {
            "swift": 5.0, "swift programming": 4.5, "swift developer": 4.5, "ios swift": 4.0,
            "swiftui": 3.5, "swift language": 4.0, "apple swift": 3.5, "swift ios": 4.0,
            "swift macos": 3.0,
        }
    ),
    (
        "Kotlin",
        {
            "kotlin": 5.0, "kotlin developer": 4.5, "kotlin android": 4.0, "kotlin programming": 4.5,
            "kotlin coroutines": 3.5, "kotlin multiplatform": 3.0, "android kotlin": 4.0,
        }
    ),
    (
        "Perl",
        {
            "perl": 5.0, "perl programming": 4.5, "perl developer": 4.5, "perl scripting": 4.0,
            "cpan": 2.5,
        }
    ),
    (
        "Shell Scripting",
        {
            "shell scripting": 5.0, "bash": 4.5, "shell script": 4.5, "bash scripting": 4.5,
            "zsh": 3.0, "shell programming": 4.0, "bash developer": 3.5, "unix shell": 3.5,
        }
    ),
    (
        "PowerShell",
        {
            "powershell": 5.0, "powershell scripting": 4.5, "powershell automation": 4.0,
            "ps1": 3.5, "powershell developer": 4.0, "azure powershell": 3.5,
        }
    ),
    (
        "Groovy",
        {
            "groovy": 5.0, "groovy programming": 4.5, "groovy developer": 4.5, "apache groovy": 4.0,
            "gradle groovy": 3.5, "groovy scripting": 3.5,
        }
    ),
    (
        "Clojure",
        {
            "clojure": 5.0, "clojure programming": 4.5, "clojure developer": 4.5,
            "clojurescript": 3.5,
        }
    ),
    (
        "Erlang",
        {
            "erlang": 5.0, "erlang programming": 4.5, "erlang developer": 4.5, "elixir erlang": 3.0,
        }
    ),
    (
        "Elixir",
        {
            "elixir": 5.0, "elixir programming": 4.5, "elixir developer": 4.5, "phoenix framework": 4.0,
            "elixir phoenix": 4.0, "elixir otp": 3.5,
        }
    ),
    (
        "Haskell",
        {
            "haskell": 5.0, "haskell programming": 4.5, "haskell developer": 4.5,
            "functional haskell": 3.5,
        }
    ),
    (
        "F#",
        {
            "f#": 5.0, "fsharp": 4.5, "f sharp": 4.5, "f# programming": 4.5, "f# developer": 4.5,
            ".net f#": 3.5,
        }
    ),
    (
        "VB.NET",
        {
            "vb.net": 5.0, "vbnet": 4.5, "visual basic": 4.0, "vb.net programming": 4.5,
            "vb.net developer": 4.5,
        }
    ),
    (
        "COBOL",
        {
            "cobol": 5.0, "cobol programming": 4.5, "cobol developer": 4.5, "mainframe cobol": 4.0,
        }
    ),
    (
        "Fortran",
        {
            "fortran": 5.0, "fortran programming": 4.5, "fortran developer": 4.5,
            "scientific computing": 3.0,
        }
    ),
    (
        "Assembly",
        {
            "assembly": 5.0, "assembly language": 4.5, "asm": 4.0, "x86 assembly": 3.5,
            "arm assembly": 3.5,
        }
    ),
    (
        "MATLAB",
        {
            "matlab": 5.0, "matlab programming": 4.5, "matlab developer": 4.5, "matlab simulink": 3.5,
            "mathematical computing": 3.0, "matlab scripting": 3.5,
        }
    ),
    (
        "R",
        {
            "r programming": 5.0, "r language": 4.5, "r developer": 4.5, "r statistical": 4.0,
            "rstudio": 3.5, "r data analysis": 4.0, "r programming language": 4.5,
        }
    ),
    (
        "Julia",
        {
            "julia": 5.0, "julia programming": 4.5, "julia developer": 4.5, "julia language": 4.5,
            "scientific julia": 3.5,
        }
    ),
    (
        "Lua",
        {
            "lua": 5.0, "lua programming": 4.5, "lua developer": 4.5, "lua scripting": 4.0,
            "lua game": 3.0,
        }
    ),
    (
        "Dart",
        {
            "dart": 5.0, "dart programming": 4.5, "dart developer": 4.5, "flutter dart": 3.5,
            "dart language": 4.0,
        }
    ),
    (
        "Objective-C",
        {
            "objective-c": 5.0, "objective c": 4.5, "objc": 4.0, "objective-c developer": 4.5,
            "ios objective-c": 4.0,
        }
    ),
    (
        "Delphi",
        {
            "delphi": 5.0, "delphi programming": 4.5, "delphi developer": 4.5, "pascal delphi": 3.5,
        }
    ),
    (
        "Pascal",
        {
            "pascal": 5.0, "pascal programming": 4.5, "pascal developer": 4.5, "object pascal": 3.5,
        }
    ),
    (
        "Ada",
        {
            "ada": 5.0, "ada programming": 4.5, "ada developer": 4.5, "ada language": 4.0,
        }
    ),
    (
        "Prolog",
        {
            "prolog": 5.0, "prolog programming": 4.5, "prolog developer": 4.5,
            "logic programming": 3.5,
        }
    ),
    (
        "Lisp",
        {
            "lisp": 5.0, "lisp programming": 4.5, "common lisp": 4.0, "scheme": 3.5,
            "clojure lisp": 2.5,
        }
    ),
    (
        "Smalltalk",
        {
            "smalltalk": 5.0, "smalltalk programming": 4.5, "smalltalk developer": 4.5,
        }
    ),
    (
        "OCaml",
        {
            "ocaml": 5.0, "ocaml programming": 4.5, "ocaml developer": 4.5, "functional ocaml": 3.5,
        }
    ),
    (
        "Racket",
        {
            "racket": 5.0, "racket programming": 4.5, "racket developer": 4.5, "racket language": 4.0,
        }
    ),
    (
        "Crystal",
        {
            "crystal": 5.0, "crystal programming": 4.5, "crystal developer": 4.5, "crystal language": 4.0,
        }
    ),
    (
        "Nim",
        {
            "nim": 5.0, "nim programming": 4.5, "nim developer": 4.5, "nim language": 4.0,
        }
    ),
    (
        "Zig",
        {
            "zig": 5.0, "zig programming": 4.5, "zig developer": 4.5, "zig language": 4.0,
        }
    ),
    (
        "V",
        {
            "v language": 5.0, "v programming": 4.5, "v developer": 4.5, "vlang": 4.0,
        }
    ),
    (
        "D",
        {
            "d programming": 5.0, "d language": 4.5, "d developer": 4.5, "dlang": 4.0,
        }
    ),
    (
        "Nix",
        {
            "nix": 5.0, "nixos": 4.0, "nix package manager": 3.5, "nix developer": 4.0,
        }
    ),
    (
        "Terraform Cloud",
        {
            "terraform cloud": 5.0, "terraform enterprise": 4.0, "terraform sentinel": 3.5,
        }
    ),
    (
        "Pulumi",
        {
            "pulumi": 5.0, "pulumi iac": 4.5, "pulumi developer": 4.5, "infrastructure pulumi": 4.0,
        }
    ),
    (
        "CloudFormation",
        {
            "cloudformation": 5.0, "aws cloudformation": 4.5, "cfn": 4.0, "cloudformation templates": 4.0,
        }
    ),
    (
        "ARM Templates",
        {
            "arm templates": 5.0, "azure resource manager": 4.5, "arm bicep": 4.0, "azure arm": 4.0,
        }
    ),
    (
        "Bicep",
        {
            "bicep": 5.0, "azure bicep": 4.5, "bicep language": 4.0, "bicep iac": 4.0,
        }
    ),
    (
        "CDK",
        {
            "cdk": 5.0, "aws cdk": 4.5, "cloud development kit": 4.5, "cdk typescript": 3.5,
            "cdk python": 3.5,
        }
    ),
    (
        "Serverless Framework",
        {
            "serverless framework": 5.0, "serverless.yml": 4.0, "serverless deploy": 3.5,
            "serverless plugin": 3.0,
        }
    ),
    (
        "SAM",
        {
            "sam": 5.0, "aws sam": 4.5, "serverless application model": 4.5, "sam template": 4.0,
        }
    ),
    (
        "Zappa",
        {
            "zappa": 5.0, "zappa python": 4.5, "zappa serverless": 4.0, "python zappa": 4.0,
        }
    ),
    (
        "Chalice",
        {
            "chalice": 5.0, "aws chalice": 4.5, "python chalice": 4.5, "serverless chalice": 4.0,
        }
    ),
    (
        "Vercel",
        {
            "vercel": 5.0, "vercel deploy": 4.0, "vercel platform": 4.0, "next.js vercel": 3.5,
        }
    ),
    (
        "Netlify",
        {
            "netlify": 5.0, "netlify deploy": 4.0, "netlify functions": 3.5, "netlify cms": 3.0,
        }
    ),
    (
        "Firebase",
        {
            "firebase": 5.0, "firebase developer": 4.5, "firebase functions": 4.0, "firebase hosting": 3.5,
            "firestore": 4.0, "firebase auth": 3.5, "firebase cloud": 3.5,
        }
    ),
    (
        "Supabase",
        {
            "supabase": 5.0, "supabase developer": 4.5, "supabase database": 4.0, "postgres supabase": 3.5,
        }
    ),
    (
        "Appwrite",
        {
            "appwrite": 5.0, "appwrite backend": 4.0, "appwrite developer": 4.5,
        }
    ),
    (
        "Hasura",
        {
            "hasura": 5.0, "hasura graphql": 4.5, "hasura developer": 4.5, "graphql hasura": 4.0,
        }
    ),
]

# Negative keywords to exclude false positives
NEGATIVE_KEYWORDS: Dict[str, Set[str]] = {
    "Java": {"javascript", "javac", "java island", "java coffee"},
    ".Net": {"avoid .net", "don't use .net", "not .net"},
    "Python": {"python snake", "monty python"},
}

# Pre-compiled regex patterns for word boundary matching
COMPILED_PATTERNS: Dict[str, Dict[str, re.Pattern]] = {}

DEFAULT_PROFILE_TYPE = "Generalist"


def canonicalize_profile_type(value: Optional[str]) -> str:
    """Normalize profile type labels to a consistent, canonical form."""
    if not value:
        return DEFAULT_PROFILE_TYPE
    
    normalized = str(value).strip()
    if not normalized:
        return DEFAULT_PROFILE_TYPE
    
    lowered = normalized.lower()
    
    # Special handling for .Net variations (net, .net, dotnet)
    # This ensures "Net", "net", ".net", "dotnet" all map to ".Net"
    if lowered in ("net", ".net", "dotnet"):
        return ".Net"
    
    # Check against canonical forms from PROFILE_TYPE_RULES
    for profile_type, _ in PROFILE_TYPE_RULES:
        if lowered == profile_type.lower():
            return profile_type
    
    # Keep known default if user explicitly passed it
    if lowered == DEFAULT_PROFILE_TYPE.lower():
        return DEFAULT_PROFILE_TYPE
    
    # If value doesn't match any known profile type, return DEFAULT_PROFILE_TYPE
    # This prevents skills from being stored as profile types
    return DEFAULT_PROFILE_TYPE


def canonicalize_profile_type_list(values: Optional[Iterable[str]]) -> List[str]:
    """Canonicalize and deduplicate a list of profile type labels."""
    if not values:
        return []
    
    canonicalized = []
    seen = set()
    for value in values:
        canonical = canonicalize_profile_type(value)
        if canonical and canonical not in seen:
            canonicalized.append(canonical)
            seen.add(canonical)
    return canonicalized

# Profile Type Compatibility Rules
# Defines which profile types can coexist (multi-profile candidates)
# This prevents incompatible combinations like "Python,Java" while allowing logical pairs
# Note: Compatibility is bidirectional - if A is compatible with B, then B is compatible with A
PROFILE_TYPE_COMPATIBILITY = {
    "Python": ["Data Science", "Full Stack", "Data Engineering", "DevOps"],
    "Java": [".Net", "Full Stack", "DevOps"],
    ".Net": ["Java", "Full Stack", "DevOps", "JavaScript"],  # Added JavaScript (Full Stack developers often have both)
    "JavaScript": ["Full Stack", "UI/UX", "Mobile Development", ".Net"],  # Added .Net (Full Stack developers often have both)
    "Full Stack": ["Python", "Java", ".Net", "JavaScript", "DevOps"],
    "Data Science": ["Python", "Data Engineering"],
    "Data Engineering": ["Python", "Data Science", "DevOps"],
    "DevOps": ["Python", "Java", ".Net", "Full Stack", "Cloud / Infra", "Data Engineering"],
    "Cloud / Infra": ["DevOps", "Data Engineering"],
    "Mobile Development": ["JavaScript", "Full Stack"],
    "UI/UX": ["JavaScript", "Full Stack"],
    "Testing / QA": ["DevOps", "Full Stack"],
    "SAP": ["ERP"],
    "ERP": ["SAP"],
    "Microsoft Power Platform": ["Low Code / No Code", "Integration / APIs"],
    "Integration / APIs": ["Microsoft Power Platform", "Full Stack"],
    "Low Code / No Code": ["Microsoft Power Platform"],
    "Salesforce": ["Integration / APIs"],
    "Database": ["Data Engineering", "DevOps", "Full Stack"],
    "Business Intelligence (BI)": ["Data Science", "Data Engineering"],
    "Cyber Security": ["DevOps", "Cloud / Infra"],
    "Business Development": [],  # Standalone profile
    "Support": [],  # Standalone profile
    # New Profile Types Compatibility
    "Go / Golang": ["Backend", "Full Stack", "Microservices", "DevOps"],
    "Ruby": ["Full Stack", "Backend"],
    "PHP": ["Full Stack", "Backend", "E-commerce", "Content Management"],
    "Rust": ["Backend", "Embedded Systems"],
    "Scala": ["Data Engineering", "Backend", "Apache Spark"],
    "C/C++": ["Embedded Systems", "Game Development"],
    "React": ["JavaScript", "Full Stack", "Frontend"],
    "Angular": ["JavaScript", "Full Stack", "Frontend"],
    "Vue.js": ["JavaScript", "Full Stack", "Frontend"],
    "Node.js": ["JavaScript", "Backend", "Full Stack"],
    "Microservices": ["Java", ".Net", "Go / Golang", "Node.js", "DevOps"],
    "Serverless": ["AWS", "Azure", "GCP", "Cloud / Infra"],
    "AWS": ["Cloud / Infra", "DevOps", "Serverless"],
    "Azure": ["Cloud / Infra", "DevOps", "Microsoft Power Platform"],
    "GCP": ["Cloud / Infra", "DevOps", "Data Engineering"],
    "Kubernetes": ["DevOps", "Docker", "Cloud / Infra"],
    "Docker": ["DevOps", "Kubernetes"],
    "Terraform": ["DevOps", "Cloud / Infra", "AWS", "Azure", "GCP"],
    "Ansible": ["DevOps", "System Administration"],
    "Jenkins": ["DevOps"],
    "GitLab CI/CD": ["DevOps"],
    "GitHub Actions": ["DevOps"],
    "MongoDB": ["Backend", "Full Stack", "Node.js"],
    "PostgreSQL": ["Backend", "Database", "Full Stack"],
    "MySQL": ["Backend", "Database", "Full Stack"],
    "Redis": ["Backend", "Full Stack"],
    "Elasticsearch": ["Data Engineering", "Backend"],
    "Apache Kafka": ["Data Engineering", "Microservices"],
    "Apache Spark": ["Data Engineering", "Scala"],
    "Hadoop": ["Data Engineering"],
    "Machine Learning": ["Python", "Data Science"],
    "Deep Learning": ["Machine Learning", "Data Science", "Python"],
    "Computer Vision": ["Deep Learning", "Machine Learning", "Python"],
    "NLP": ["Deep Learning", "Machine Learning", "Python"],
    "Blockchain": ["Web3", "Backend"],
    "Web3": ["Blockchain", "JavaScript", "Backend"],
    "IoT": ["Embedded Systems", "C/C++", "Python"],
    "Embedded Systems": ["C/C++", "IoT"],
    "Game Development": ["C/C++", ".Net", "Unity"],
    "AR/VR": ["Game Development"],
    "FinTech": ["Backend", "Java", ".Net", "Python"],
    "Healthcare IT": ["Backend", "Integration / APIs"],
    "E-commerce": ["Full Stack", "PHP", "JavaScript"],
    "Content Management": ["PHP", "Full Stack"],
    "Video Streaming": ["Backend", "Python", "Node.js"],
    "Network Engineering": ["System Administration"],
    "System Administration": ["Network Engineering", "DevOps", "Linux"],
    "GraphQL": ["JavaScript", "Node.js", "Backend", "Full Stack"],
    "TypeScript": ["JavaScript", "Angular", "React", "Node.js"],
    "Linux": ["System Administration", "DevOps", "Backend"],
    # Additional 50 Profile Types Compatibility
    "Swift": ["iOS", "Mobile Development"],
    "Kotlin": ["Android", "Mobile Development"],
    "Perl": ["Backend", "Scripting"],
    "Shell Scripting": ["Linux", "System Administration", "DevOps"],
    "PowerShell": ["Azure", "System Administration", "Windows"],
    "Groovy": ["Java", "Gradle", "Backend"],
    "Clojure": ["Backend", "Functional Programming"],
    "Erlang": ["Backend", "Elixir"],
    "Elixir": ["Backend", "Erlang", "Phoenix"],
    "Haskell": ["Backend", "Functional Programming"],
    "F#": [".Net", "Backend"],
    "VB.NET": [".Net", "Backend"],
    "COBOL": ["Mainframe", "Legacy Systems"],
    "Fortran": ["Scientific Computing", "HPC"],
    "Assembly": ["Embedded Systems", "Systems Programming"],
    "MATLAB": ["Scientific Computing", "Data Science"],
    "R": ["Data Science", "Statistics", "Business Intelligence (BI)"],
    "Julia": ["Data Science", "Scientific Computing"],
    "Lua": ["Game Development", "Scripting"],
    "Dart": ["Flutter", "Mobile Development"],
    "Objective-C": ["iOS", "Mobile Development"],
    "Delphi": ["Desktop Development", "Windows"],
    "Pascal": ["Desktop Development", "Legacy Systems"],
    "Ada": ["Embedded Systems", "Safety-Critical"],
    "Prolog": ["AI", "Logic Programming"],
    "Lisp": ["Functional Programming", "AI"],
    "Smalltalk": ["Object-Oriented", "Legacy Systems"],
    "OCaml": ["Functional Programming", "Backend"],
    "Racket": ["Functional Programming", "Education"],
    "Crystal": ["Backend", "Ruby"],
    "Nim": ["Systems Programming", "Backend"],
    "Zig": ["Systems Programming", "Embedded Systems"],
    "V": ["Systems Programming", "Backend"],
    "D": ["Systems Programming", "Backend"],
    "Nix": ["DevOps", "Package Management"],
    "Terraform Cloud": ["Terraform", "DevOps", "Cloud / Infra"],
    "Pulumi": ["DevOps", "Cloud / Infra", "Infrastructure as Code"],
    "CloudFormation": ["AWS", "DevOps", "Cloud / Infra"],
    "ARM Templates": ["Azure", "DevOps", "Cloud / Infra"],
    "Bicep": ["Azure", "DevOps", "Cloud / Infra"],
    "CDK": ["AWS", "DevOps", "TypeScript", "Python"],
    "Serverless Framework": ["Serverless", "AWS", "Node.js"],
    "SAM": ["AWS", "Serverless", "Cloud / Infra"],
    "Zappa": ["Python", "Serverless", "AWS"],
    "Chalice": ["Python", "Serverless", "AWS"],
    "Vercel": ["Next.js", "Serverless", "Frontend"],
    "Netlify": ["Frontend", "Serverless", "JAMstack"],
    "Firebase": ["Backend", "Mobile Development", "Serverless"],
    "Supabase": ["PostgreSQL", "Backend", "Full Stack"],
    "Appwrite": ["Backend", "Full Stack", "Mobile Development"],
    "Hasura": ["GraphQL", "Backend", "PostgreSQL"],
}

def are_profile_types_compatible(profile1: str, profile2: str) -> bool:
    """
    Check if two profile types are compatible (can coexist in multi-profile candidate).
    
    Uses bidirectional checking - if A is compatible with B, then B is compatible with A.
    This ensures consistency regardless of which profile is checked first.
    
    Args:
        profile1: First profile type
        profile2: Second profile type
        
    Returns:
        True if profiles are compatible, False otherwise
    """
    profile1 = canonicalize_profile_type(profile1)
    profile2 = canonicalize_profile_type(profile2)
    
    if profile1 == profile2:
        return True
    
    # Bidirectional check: A compatible with B OR B compatible with A
    # This ensures consistency regardless of order
    compatible_list_1 = PROFILE_TYPE_COMPATIBILITY.get(profile1, [])
    compatible_list_2 = PROFILE_TYPE_COMPATIBILITY.get(profile2, [])
    
    # Check both directions for compatibility
    return (profile2 in compatible_list_1) or (profile1 in compatible_list_2)


def _compile_keyword_patterns():
    """Pre-compile regex patterns for all keywords with word boundaries."""
    global COMPILED_PATTERNS
    if COMPILED_PATTERNS:
        return
    
    for profile_type, keyword_weights in PROFILE_TYPE_RULES_ENHANCED:
        COMPILED_PATTERNS[profile_type] = {}
        for keyword in keyword_weights.keys():
            escaped = re.escape(keyword.lower())
            if ' ' in keyword:
                pattern = r'\b' + escaped.replace(r'\ ', r'\s+') + r'\b'
            else:
                pattern = r'\b' + escaped + r'\b'
            COMPILED_PATTERNS[profile_type][keyword] = re.compile(pattern, re.IGNORECASE)

_compile_keyword_patterns()

def _normalize_text_blob(*parts: str) -> str:
    """Lower-case, concatenated blob for keyword detection."""
    normalized_parts = []
    for part in parts:
        if not part:
            continue
        normalized_parts.append(str(part).lower())
    return " ".join(normalized_parts)


def _count_keyword_matches(keyword: str, text: str, profile_type: str) -> int:
    """Count keyword matches using word boundaries."""
    pattern = COMPILED_PATTERNS.get(profile_type, {}).get(keyword)
    if not pattern:
        if profile_type == "Python" and keyword == "python":
            logger.warning(f"DEBUG _count_keyword_matches: Pattern not found for Python/python. "
                          f"COMPILED_PATTERNS has 'Python' key: {'Python' in COMPILED_PATTERNS}, "
                          f"Python dict has 'python' key: {'python' in COMPILED_PATTERNS.get('Python', {})}")
        return 0
    matches = pattern.findall(text)
    count = len(matches)
    if profile_type == "Python" and keyword == "python":
        logger.info(f"DEBUG _count_keyword_matches: pattern={pattern.pattern}, matches={matches}, count={count}")
    return count

def _has_negative_context(keyword: str, text: str, profile_type: str) -> bool:
    """Check if keyword appears in negative context."""
    negative_patterns = NEGATIVE_KEYWORDS.get(profile_type, set())
    pattern = COMPILED_PATTERNS.get(profile_type, {}).get(keyword)
    if not pattern:
        return False
    
    # Compile negative indicator patterns with word boundaries to avoid false positives
    # e.g., "not" should match "not python" but NOT "notebook"
    negative_indicator_patterns = [
        re.compile(r"\bdon'?t\b", re.IGNORECASE),  # Matches "don't" or "dont" as whole words
        re.compile(r"\bnot\b", re.IGNORECASE),  # Matches "not" as whole word (not "notebook")
        re.compile(r"\bavoid\b", re.IGNORECASE),
        re.compile(r"\bnever\b", re.IGNORECASE),
        re.compile(r"\bno experience\b", re.IGNORECASE),
        re.compile(r"\bnot familiar\b", re.IGNORECASE),
        re.compile(r"\bunfamiliar\b", re.IGNORECASE),
        re.compile(r"\bdon'?t know\b", re.IGNORECASE),  # Matches "don't know" or "dont know"
    ]
    
    for match in pattern.finditer(text):
        start, end = match.span()
        context = text[max(0, start-50):min(len(text), end+50)]
        
        # Check each negative indicator pattern
        for neg_pattern in negative_indicator_patterns:
            if neg_pattern.search(context):
                return True
    return False

def detect_profile_types_from_text(*parts: str) -> List[str]:
    """
    Return a prioritized list of profile types detected inside the provided text parts.
    Multiple profile types may apply (e.g., Full Stack + JavaScript).
    Uses word boundary matching to avoid false positives.
    """
    text_blob = _normalize_text_blob(*parts)
    if not text_blob:
        return []
    
    matches = []
    for profile_type, keywords in PROFILE_TYPE_RULES:
        for keyword in keywords:
            if _count_keyword_matches(keyword, text_blob, profile_type) > 0:
                matches.append(profile_type)
                break
    return matches


def determine_profile_types_enhanced(
    primary_skills: str = "",
    secondary_skills: str = "",
    resume_text: str = "",
    ai_client=None,
    ai_model: str = None,
    min_confidence: float = 0.01-0.1,  # Increased from 0.3 to 0.4 for stricter filtering
    equal_score_threshold: float = 0.15,
    max_profiles: int = 2  # Reduced from 3 to 2 to prevent too many profiles
) -> Tuple[List[str], float, Dict[str, Any]]:
    """
    Enhanced profile type detection with multi-profile support.
    
    Args:
        primary_skills: Comma-separated primary skills
        secondary_skills: Comma-separated secondary skills
        resume_text: Full resume text content
        ai_client: Optional AI/LLM client
        ai_model: Optional AI model name
        min_confidence: Minimum confidence threshold (0.0-1.0)
        equal_score_threshold: Score difference ratio for equal profiles (0.0-1.0)
        max_profiles: Maximum number of profiles to return
        
    Returns:
        (profile_types, overall_confidence, metadata)
    """
    metadata = {'method': 'keyword', 'scores': {}, 'matched_keywords': {}}
    
    # AI/LLM detection disabled - always use keyword-based detection for consistency and accuracy
    # Keyword-based detection provides more reliable and consistent results for profile type classification
    
    # Keyword-based detection
    text_blob = _normalize_text_blob(primary_skills, secondary_skills, resume_text)
    if not text_blob:
        return ([DEFAULT_PROFILE_TYPE], 0.0, metadata)
    
    profile_scores = _calculate_normalized_scores(text_blob, primary_skills, secondary_skills, resume_text)
    
    if not profile_scores:
        return ([DEFAULT_PROFILE_TYPE], 0.0, metadata)
    
    # Store metadata
    for ps in profile_scores[:max_profiles]:
        metadata['scores'][ps.profile_type] = {
            'normalized': ps.normalized_score,
            'raw': ps.raw_score,
            'confidence': ps.confidence
        }
        metadata['matched_keywords'][ps.profile_type] = ps.matched_keywords
    
    # Filter by confidence
    valid_scores = [ps for ps in profile_scores if ps.confidence >= min_confidence]
    if not valid_scores:
        return ([DEFAULT_PROFILE_TYPE], 0.0, metadata)
    
    # Multi-profile logic with adaptive requirements based on confidence and score strength
    top_score = valid_scores[0]
    equal_profiles = [top_score.profile_type]
    top_normalized = top_score.normalized_score
    top_confidence = top_score.confidence
    
    # Log detailed scoring for debugging
    logger.info(f"Profile type scoring details:")
    for i, ps in enumerate(valid_scores[:max_profiles]):
        score_diff = top_normalized - ps.normalized_score if i > 0 else 0.0
        score_diff_ratio = (score_diff / top_normalized * 100) if top_normalized > 0 and i > 0 else 0.0
        score_ratio = (ps.normalized_score / top_normalized * 100) if top_normalized > 0 else 0.0
        logger.info(
            f"  {i+1}. {ps.profile_type}: "
            f"normalized={ps.normalized_score:.4f} ({score_ratio:.1f}% of top), "
            f"raw={ps.raw_score:.2f}, "
            f"confidence={ps.confidence:.3f}, "
            f"diff={score_diff_ratio:.1f}%, "
            f"keywords={ps.matched_keywords[:3]}"
        )
    
    # Adaptive multi-profile inclusion criteria based on top score strength and confidence
    # Tier 1: High confidence (>=0.75) or strong top score (>=0.7) → Single profile preferred
    # Tier 2: Moderate confidence (0.6-0.75) and moderate score (0.5-0.7) → Allow multi-profile with strict criteria
    # Tier 3: Low confidence (<0.6) or weak score (<0.5) → Single profile only (most reliable)
    
    if top_confidence >= 0.75 or top_normalized >= 0.7:
        # Tier 1: High confidence/strong score - very strict for second profile
        tier = "Tier 1 (High Confidence/Strong Score)"
        min_keywords_required = 5  # Increased from 4 to 5 - require more keywords for higher accuracy
        min_score_ratio = 0.75  # Increased from 0.65 to 0.75 - second must be at least 75% of top
        min_raw_score_for_inclusion = 30.0  # Increased from 25.0 to 30.0 - higher raw score required
        dominant_score_threshold = 0.15  # Increased from 0.01 to 0.15 (15%) - exclude if top is >15% higher
        min_second_confidence = 0.70  # Increased from 0.65 to 0.70 - higher confidence required
    elif top_confidence >= 0.6 and top_normalized >= 0.5:
        # Tier 2: Moderate confidence/score - moderate strictness
        tier = "Tier 2 (Moderate Confidence/Score)"
        min_keywords_required = 4  # Increased from 3 to 4 - require more keywords for higher accuracy
        min_score_ratio = 0.65  # Increased from 0.55 to 0.65 - second must be at least 65% of top
        min_raw_score_for_inclusion = 25.0  # Increased from 20.0 to 25.0 - higher raw score required
        dominant_score_threshold = 0.10  # Increased from 0.01 to 0.10 (10%) - exclude if top is >10% higher
        min_second_confidence = 0.60  # Increased from 0.55 to 0.60 - higher confidence required
    else:
        # Tier 3: Low confidence/weak score - single profile only
        tier = "Tier 3 (Low Confidence/Weak Score)"
        min_keywords_required = 999  # Effectively disables multi-profile
        min_score_ratio = 1.0
        min_raw_score_for_inclusion = 999.0
        dominant_score_threshold = 0.0
        min_second_confidence = 1.0
    
    logger.info(f"Using {tier} criteria for multi-profile inclusion")
    
    for score in valid_scores[1:max_profiles]:
        score_diff_ratio = (top_normalized - score.normalized_score) / top_normalized if top_normalized > 0 else 1.0
        score_ratio = score.normalized_score / top_normalized if top_normalized > 0 else 0.0
        
        # Count only real keywords (exclude phrase matches)
        real_keywords = [kw for kw in score.matched_keywords if not kw.startswith("phrase_match_")]
        keyword_count = len(real_keywords)
        
        # Check if top score is significantly dominant
        is_top_dominant = score_diff_ratio > dominant_score_threshold
        
        # Check profile type compatibility
        is_compatible = are_profile_types_compatible(top_score.profile_type, score.profile_type)
        
        # Check confidence requirement for second profile
        has_sufficient_confidence = score.confidence >= min_second_confidence
        
        # Check ALL strict inclusion criteria
        meets_diff_threshold = score_diff_ratio <= equal_score_threshold
        meets_min_ratio = score_ratio >= min_score_ratio
        has_significant_raw_score = score.raw_score >= min_raw_score_for_inclusion
        has_min_keywords = keyword_count >= min_keywords_required
        
        # Include ONLY if ALL conditions are met:
        # 1. Top score is not significantly dominant
        # 2. Profile types are compatible
        # 3. Second profile has sufficient confidence
        # 4. All other criteria met
        if (not is_top_dominant and is_compatible and has_sufficient_confidence and 
            meets_diff_threshold and meets_min_ratio and has_significant_raw_score and has_min_keywords):
            equal_profiles.append(score.profile_type)
            logger.info(
                f"  → Including {score.profile_type} "
                f"(keywords={keyword_count}, raw={score.raw_score:.1f}, "
                f"ratio={score_ratio*100:.1f}%, diff={score_diff_ratio*100:.1f}%, "
                f"confidence={score.confidence:.3f}, compatible={is_compatible})"
            )
        else:
            reason = []
            if is_top_dominant:
                reason.append(f"top score is dominant (diff {score_diff_ratio*100:.1f}% > {dominant_score_threshold*100:.1f}%)")
            if not is_compatible:
                reason.append(f"profile types incompatible ({top_score.profile_type} + {score.profile_type})")
            if not has_sufficient_confidence:
                reason.append(f"confidence {score.confidence:.3f} < {min_second_confidence:.3f}")
            if not has_min_keywords:
                reason.append(f"keywords {keyword_count} < {min_keywords_required}")
            if not has_significant_raw_score:
                reason.append(f"raw score {score.raw_score:.1f} < {min_raw_score_for_inclusion}")
            if not meets_min_ratio:
                reason.append(f"ratio {score_ratio*100:.1f}% < {min_score_ratio*100:.1f}%")
            if not meets_diff_threshold:
                reason.append(f"diff {score_diff_ratio*100:.1f}% > {equal_score_threshold*100:.1f}%")
            logger.info(f"  → Excluding {score.profile_type} ({', '.join(reason)})")
            break
    
    overall_confidence = sum(ps.confidence for ps in valid_scores[:len(equal_profiles)]) / len(equal_profiles)
    # Canonicalize and deduplicate profile types (remove duplicates while preserving order)
    canonical_profiles = []
    seen = set()
    for pt in equal_profiles:
        canonical = canonicalize_profile_type(pt)
        if canonical and canonical not in seen:
            canonical_profiles.append(canonical)
            seen.add(canonical)
    
    logger.info(
        f"Detected profiles: {canonical_profiles} "
        f"(scores: {[f'{ps.normalized_score:.2f}' for ps in valid_scores[:len(equal_profiles)]]}, "
        f"confidence: {overall_confidence:.2f})"
    )
    
    return (canonical_profiles, overall_confidence, metadata)

def determine_primary_profile_type(primary_skills: str = "", secondary_skills: str = "", resume_text: str = "", ai_client=None, ai_model: str = None) -> str:
    """
    Determine the canonical profile type (backward compatible - returns single profile).
    For multi-profile support, use determine_profile_types_enhanced().
    
    Args:
        primary_skills: Comma-separated primary skills
        secondary_skills: Comma-separated secondary skills  
        resume_text: Full resume text content
        ai_client: Optional AI/LLM client for intelligent analysis
        ai_model: Optional AI model name
        
    Returns:
        Canonical profile type string (comma-separated if multiple)
    """
    profile_types, confidence, _ = determine_profile_types_enhanced(
        primary_skills, secondary_skills, resume_text, ai_client, ai_model
    )
    # Return comma-separated string for backward compatibility
    return ", ".join(profile_types) if len(profile_types) > 1 else profile_types[0] if profile_types else DEFAULT_PROFILE_TYPE


def _determine_profile_type_with_llm_enhanced(
    primary_skills: str,
    secondary_skills: str,
    resume_text: str,
    ai_client,
    ai_model: str
) -> Optional[Tuple[List[str], float]]:
    """Enhanced LLM detection with multi-profile support."""
    skills_context = f"Primary Skills: {primary_skills}\nSecondary Skills: {secondary_skills}"
    resume_snippet = resume_text[:4000] if len(resume_text) > 4000 else resume_text
    profile_types_list = [pt for pt, _ in PROFILE_TYPE_RULES_ENHANCED] + [DEFAULT_PROFILE_TYPE]
    
    prompt = f"""Analyze the resume and determine profile types. Return comma-separated if multiple apply.

{skills_context}

Resume Content:
{resume_snippet}

Available Profile Types: {', '.join(profile_types_list)}

Examples:
- "Java, Spring Boot, Hibernate" → "Java"
- "C#, ASP.NET, Entity Framework, Java, Spring" → ".Net, Java"
- "React, Node.js, MongoDB" → "Full Stack"

Return format: Comma-separated profile types (e.g., ".Net, Java" or "Python")
If uncertain, return "{DEFAULT_PROFILE_TYPE}".
"""
    
    try:
        response = ai_client.chat.completions.create(
            model=ai_model,
            messages=[
                {"role": "system", "content": "Expert technical recruiter identifying profile types."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        result = response.choices[0].message.content.strip()
        result = result.strip('"\'').strip('.').strip()
        
        profiles = [p.strip() for p in result.split(',')]
        # Filter to only valid profile types (canonicalize returns DEFAULT_PROFILE_TYPE for invalid ones)
        canonical_profiles = [
            pt for p in profiles if p 
            for pt in [canonicalize_profile_type(p)]
            if pt != DEFAULT_PROFILE_TYPE
        ]
        
        # Limit LLM output to max 2 profiles to prevent too many profiles
        if canonical_profiles:
            limited_profiles = canonical_profiles[:2]  # Strict limit of 2 profiles
            return (limited_profiles, 0.85)
        
    except Exception as e:
        logger.error(f"LLM profile type determination failed: {e}")
    
    return None

def get_all_profile_type_scores(
    primary_skills: str = "",
    secondary_skills: str = "",
    resume_text: str = ""
) -> Dict[str, float]:
    """
    Get raw scores for ALL profile types (not just top ones).
    This is used to store scores in candidate_profile_scores table.
    
    Args:
        primary_skills: Comma-separated primary skills
        secondary_skills: Comma-separated secondary skills
        resume_text: Full resume text content
        
    Returns:
        Dictionary mapping profile_type -> raw_score (actual calculated values like 12, 25, 100)
        Includes all profile types, even if score is 0.0
    """
    # Normalize text
    text_blob = _normalize_text_blob(primary_skills, secondary_skills, resume_text)
    if not text_blob:
        # Return all zeros if no text
        return {pt: 0.0 for pt, _ in PROFILE_TYPE_RULES_ENHANCED}
    
    # Calculate scores for all profile types
    profile_scores = _calculate_normalized_scores(text_blob, primary_skills, secondary_skills, resume_text)
    
    # Create dictionary with all profile types
    all_scores = {pt: 0.0 for pt, _ in PROFILE_TYPE_RULES_ENHANCED}
    
    # Update with calculated raw scores (not normalized)
    for ps in profile_scores:
        all_scores[ps.profile_type] = round(ps.raw_score, 2)  # Round to 2 decimal places for raw scores
    
    # Debug logging for Python score
    python_score = all_scores.get("Python", 0.0)
    python_in_scores = any(ps.profile_type == "Python" for ps in profile_scores)
    logger.info(f"DEBUG get_all_profile_type_scores: Python score={python_score}, "
              f"Python in profile_scores list={python_in_scores}, "
              f"primary_skills='{primary_skills[:100] if primary_skills else 'EMPTY'}', "
              f"text_blob length={len(text_blob)}")
    if python_score == 0.0 and "python" in text_blob.lower():
        logger.warning(f"DEBUG Python score is 0 but 'python' found in text_blob! "
                      f"profile_scores list has {len(profile_scores)} items")
        for ps in profile_scores[:5]:
            logger.info(f"  Top scores: {ps.profile_type}={ps.raw_score}")
    
    return all_scores

def _determine_profile_type_with_llm(primary_skills: str, secondary_skills: str, resume_text: str, ai_client, ai_model: str) -> str:
    """
    Use LLM to analyze overall resume content and determine profile type.
    """
    
    # Prepare context for LLM analysis
    skills_context = f"Primary Skills: {primary_skills}\nSecondary Skills: {secondary_skills}"
    resume_snippet = resume_text[:4000] if len(resume_text) > 4000 else resume_text  # Limit text for token efficiency
    
    profile_types_list = [pt for pt, _ in PROFILE_TYPE_RULES] + [DEFAULT_PROFILE_TYPE]
    
    prompt = f"""Analyze the following resume content and skill set to determine the candidate's primary profile type.

{skills_context}

Resume Content (snippet):
{resume_snippet}

Available Profile Types:
{', '.join(profile_types_list)}

Instructions:
1. Analyze the overall resume content, skills, experience, and context
2. Consider the dominant technology stack, frameworks, and tools mentioned
3. Identify the PRIMARY profile type that best describes this candidate
4. Consider skill weights - if candidate has C#, ASP.NET, .NET Core, ADO.NET, Entity Framework - they are clearly a .NET developer, not Java
5. Return ONLY the profile type name (one of the available types), nothing else

Return format: Just the profile type name (e.g., ".Net", "Java", "Python", etc.)
If uncertain, return "{DEFAULT_PROFILE_TYPE}".
"""
    
    try:
        response = ai_client.chat.completions.create(
            model=ai_model,
            messages=[
                {"role": "system", "content": "You are an expert technical recruiter who accurately identifies candidate profile types based on resume analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent results
            max_tokens=50
        )
        
        result = response.choices[0].message.content.strip()
        # Clean up the response - remove quotes, periods, etc.
        result = result.strip('"\'').strip('.').strip()
        
        logger.info(f"LLM determined profile type: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in LLM profile type determination: {e}")
        return None


class ProfileScore(NamedTuple):
    """Profile type scoring result."""
    profile_type: str
    raw_score: float
    normalized_score: float
    confidence: float
    matched_keywords: List[str]
    keyword_details: List[Tuple[str, float, str]]  # (keyword, score, location)

def _extract_first_lines_of_skills(primary_skills: str, num_lines: int = 2) -> str:
    """Extract first 1-2 lines of technical skills for higher weightage."""
    if not primary_skills:
        return ""
    
    # Split by comma and take first few skills (assuming comma-separated)
    skills_list = [s.strip() for s in primary_skills.split(',')]
    if len(skills_list) <= num_lines:
        return primary_skills
    
    # Return first 1-2 skills
    return ', '.join(skills_list[:num_lines])

def _detect_phrases(text: str) -> Dict[str, float]:
    """Detect specific phrases that indicate profile types with high confidence."""
    text_lower = text.lower()
    phrase_scores = {}
    
    # Python phrases
    python_phrases = [
        (r"with solid foundation.*?python", 8.0),
        (r"strong experience.*?python", 8.0),
        (r"solid foundation in python", 8.0),
        (r"strong experience in python", 8.0),
        (r"expertise.*?python", 6.0),
        (r"proficient.*?python", 6.0),
    ]
    
    for pattern, weight in python_phrases:
        if re.search(pattern, text_lower, re.IGNORECASE):
            phrase_scores['Python'] = max(phrase_scores.get('Python', 0.0), weight)
    
    # Java phrases
    java_phrases = [
        (r"with solid foundation.*?java", 8.0),
        (r"strong experience.*?java", 8.0),
        (r"solid foundation in java", 8.0),
        (r"strong experience in java", 8.0),
        (r"expertise.*?java", 6.0),
        (r"proficient.*?java", 6.0),
    ]
    
    for pattern, weight in java_phrases:
        if re.search(pattern, text_lower, re.IGNORECASE):
            phrase_scores['Java'] = max(phrase_scores.get('Java', 0.0), weight)
    
    return phrase_scores

def _check_business_development(text: str) -> bool:
    """
    Check if profile indicates Business Development role using safe,
    word-boundary-aware patterns. Requires multiple high-confidence signals
    to prevent false positives from technical terms like "client acquisition"
    in software contexts.
    
    Returns True only if:
    - "business development" (exact phrase) is found, OR
    - At least 2 BD-specific keywords are found (excluding ambiguous ones)
    """
    if not text:
        return False

    # High-confidence patterns (exact BD phrases)
    high_confidence_patterns = [
        r"\bbusiness\s+development\b",
        r"\bbusiness\s+development\s+executive\b",
        r"\bbusiness\s+development\s+manager\b",
        r"\bbusiness\s+dev\b",
    ]
    
    # Medium-confidence patterns (require context - count these)
    medium_confidence_patterns = [
        r"\bb2b\s+sales\b",
        r"\bpartnership\s+development\b",
        r"\bstrategic\s+partnerships\b",
        r"\baccount\s+development\b",
    ]
    
    # Low-confidence patterns (ambiguous - only count if other signals present)
    # Excluded: "client acquisition", "market expansion" - too common in technical contexts
    # Excluded: "bd", "bde" - too short, can match in other words
    
    text_lower = text.lower()
    
    # Check high-confidence patterns first (if found, definitely BD)
    for pattern in high_confidence_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    # Require at least 2 medium-confidence patterns to avoid false positives
    medium_matches = sum(1 for pattern in medium_confidence_patterns 
                         if re.search(pattern, text_lower, re.IGNORECASE))
    
    return medium_matches >= 2

def _calculate_normalized_scores(
    text_blob: str,
    primary_skills: str,
    secondary_skills: str,
    resume_text: str
) -> List[ProfileScore]:
    """Calculate normalized scores with confidence for all profile types."""
    profile_scores = []
    primary_lower = primary_skills.lower()
    secondary_lower = secondary_skills.lower()
    
    # Extract first 1-2 lines of skills for higher weightage
    first_lines_skills = _extract_first_lines_of_skills(primary_skills, num_lines=2)
    first_lines_lower = first_lines_skills.lower()
    
    # Detect specific phrases
    phrase_scores = _detect_phrases(resume_text + " " + primary_skills)
    
    # Check for Business Development (for logging/debugging, but don't short-circuit)
    # All profiles will be scored normally, allowing BD to compete with other profiles
    is_business_dev = _check_business_development(resume_text + " " + primary_skills)
    if is_business_dev:
        logger.info("Business Development signals detected, but allowing all profiles to compete")
    
    for profile_type, keyword_weights in PROFILE_TYPE_RULES_ENHANCED:
        
        raw_score = 0.0
        matched_keywords = []
        keyword_details = []
        
        for keyword, base_weight in keyword_weights.items():
            # Debug logging for Python profile type
            if profile_type == "Python" and keyword == "python":
                pattern = COMPILED_PATTERNS.get(profile_type, {}).get(keyword)
                logger.info(f"DEBUG Python matching: keyword='{keyword}', pattern exists={pattern is not None}")
                if pattern:
                    logger.info(f"DEBUG Python pattern: {pattern.pattern}")
                logger.info(f"DEBUG Python text_blob contains 'python': {'python' in text_blob}")
                logger.info(f"DEBUG Python text_blob sample: {text_blob[:200]}")
            
            if _has_negative_context(keyword, text_blob, profile_type):
                if profile_type == "Python" and keyword == "python":
                    logger.warning(f"DEBUG Python: Skipped due to negative context")
                continue
            
            count = _count_keyword_matches(keyword, text_blob, profile_type)
            if profile_type == "Python" and keyword == "python":
                logger.info(f"DEBUG Python: count={count}, base_weight={base_weight}")
            if count == 0:
                if profile_type == "Python" and keyword == "python":
                    logger.warning(f"DEBUG Python: count is 0, skipping keyword")
                continue
            
            location_multiplier = 1.0
            # Check if keyword is in first 1-2 lines of skills (highest priority)
            if keyword.lower() in first_lines_lower:
                location_multiplier = 5.0  # Highest weight for first lines
            elif keyword.lower() in primary_lower:
                location_multiplier = 3.0
            elif keyword.lower() in secondary_lower:
                location_multiplier = 2.0
            
            if profile_type == "Python" and keyword == "python":
                logger.info(f"DEBUG Python location: first_lines_lower contains='{'python' in first_lines_lower}', "
                          f"primary_lower contains='{'python' in primary_lower}', "
                          f"location_multiplier={location_multiplier}")
                logger.info(f"DEBUG Python first_lines: '{first_lines_lower[:100]}'")
                logger.info(f"DEBUG Python primary_lower sample: '{primary_lower[:100]}'")
            
            keyword_score = count * base_weight * location_multiplier
            raw_score += keyword_score
            
            if profile_type == "Python" and keyword == "python":
                logger.info(f"DEBUG Python: keyword_score={keyword_score}, raw_score so far={raw_score}")
            
            matched_keywords.append(keyword)
            location = ("first_lines" if location_multiplier == 5.0
                       else "primary" if location_multiplier == 3.0 
                       else "secondary" if location_multiplier == 2.0 
                       else "resume")
            keyword_details.append((keyword, keyword_score, location))
        
        # Add phrase-based bonus
        if profile_type in phrase_scores:
            phrase_bonus = phrase_scores[profile_type]
            raw_score += phrase_bonus
            matched_keywords.append(f"phrase_match_{profile_type.lower()}")
            keyword_details.append((f"phrase_match", phrase_bonus, "resume"))
        
        if raw_score > 0:
            # Max possible uses 5.0 multiplier to account for first_lines weightage
            max_possible = sum(keyword_weights.values()) * 5.0
            normalized_score = min(1.0, raw_score / max_possible) if max_possible > 0 else 0.0
            confidence = _calculate_confidence(
                normalized_score, len(matched_keywords), len(keyword_weights), keyword_details
            )
            profile_scores.append(ProfileScore(
                profile_type=profile_type,
                raw_score=raw_score,
                normalized_score=normalized_score,
                confidence=confidence,
                matched_keywords=matched_keywords,
                keyword_details=keyword_details
            ))
        elif profile_type == "Python":
            logger.warning(f"DEBUG Python: raw_score is 0, NOT added to profile_scores. "
                          f"Matched keywords count: {len(matched_keywords)}, "
                          f"text_blob length: {len(text_blob)}")
    
    return sorted(profile_scores, key=lambda x: x.normalized_score, reverse=True)

def _calculate_confidence(
    normalized_score: float,
    matched_count: int,
    total_keywords: int,
    keyword_details: List[Tuple[str, float, str]]
) -> float:
    """Calculate confidence score (0.0 to 1.0)."""
    base_confidence = normalized_score
    coverage_ratio = matched_count / total_keywords if total_keywords > 0 else 0
    coverage_bonus = min(0.2, coverage_ratio * 0.2)
    
    primary_bonus = 0.0
    for _, score, location in keyword_details:
        if location == "primary" and score > 5.0:
            primary_bonus += 0.1
    
    confidence = min(1.0, base_confidence + coverage_bonus + min(0.1, primary_bonus))
    return round(confidence, 3)

def _determine_profile_type_with_keywords(primary_skills: str, secondary_skills: str, resume_text: str) -> str:
    """
    Fallback: Weighted keyword-based profile type detection (legacy method).
    Uses enhanced scoring but returns single profile for backward compatibility.
    """
    profile_types, confidence, _ = determine_profile_types_enhanced(
        primary_skills, secondary_skills, resume_text, None, None
    )
    return profile_types[0] if profile_types else DEFAULT_PROFILE_TYPE

def format_profile_types_for_storage(profile_types: List[str]) -> str:
    """Format profile types for database storage (comma-separated)."""
    if not profile_types:
        return DEFAULT_PROFILE_TYPE
    
    # Filter to only valid profile types (exclude DEFAULT_PROFILE_TYPE and invalid entries)
    canonical = [
        pt for profile_type in profile_types
        for pt in [canonicalize_profile_type(profile_type)]
        if pt != DEFAULT_PROFILE_TYPE
    ]
    
    if not canonical:
        return DEFAULT_PROFILE_TYPE
    
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for pt in canonical:
        if pt not in seen:
            seen.add(pt)
            unique.append(pt)
    
    # Use comma without space for MySQL FIND_IN_SET compatibility
    # Stored format: "Microsoft Power Platform,Integration / APIs"
    return ",".join(unique)


def infer_profile_type_from_requirements(required_skills: List[str], jd_text: str = "") -> List[str]:
    """
    Infer one or more target profile types from job requirements / JD text.
    """
    skill_blob = ", ".join(required_skills or [])
    combined = _normalize_text_blob(skill_blob, jd_text)
    matches = detect_profile_types_from_text(combined)
    # Deduplicate while preserving order
    seen = set()
    ordered_matches = []
    for match in matches:
        if match not in seen:
            ordered_matches.append(match)
            seen.add(match)
    return ordered_matches

