const path = require("path");
const child_process = require("child_process");
const fs = require("fs-extra");
const inquirer = require("@jyeontu/j-inquirer");
const crypto = require("crypto");

function initBuildConfig() {
  if (fs.existsSync("./buildConfig.json")) return;
  fs.createFileSync("./buildConfig.json");
  const data = {
    HASH: {},
    PACKAGE: {
      checkKeyList: ["dependencies", "devDependencies"],
    },
  };
  const jsonData = JSON.stringify(data, null, 2);
  fs.writeFileSync("./buildConfig.json", jsonData, { encoding: "utf8" });
}
initBuildConfig();

const buildConfig = require("./buildConfig.json");
const { HASH, PACKAGE } = buildConfig;

function hashFolder(folderPath) {
  const hash = crypto.createHash("sha256");
  const files = fs.readdirSync(folderPath);

  files.forEach((file) => {
    const filePath = path.join(folderPath, file);
    const stats = fs.statSync(filePath);

    if (stats.isFile()) {
      const data = fs.readFileSync(filePath);
      hash.update(data);
    } else if (stats.isDirectory()) {
      hash.update(hashFolder(filePath));
    }
  });

  return hash.digest("hex");
}

async function doInquirer(options) {
  const answers = await new inquirer(options).prompt();
  return answers;
}
function updateConfigPackage(options) {
  const dir = __dirname + "\\";
  const cwd = options.cwd;
  const cwdKey = cwd.replace(dir, "") || "./";
  const checkKeyList = PACKAGE.checkKeyList;
  const packagePath = path.join(cwd, "package.json");
  const packageJson = require(packagePath);
  const configJson = {};
  for (const checkKey of checkKeyList) {
    const obj = packageJson[checkKey];
    configJson[checkKey] = obj;
    PACKAGE[cwdKey] = configJson;
    buildConfig.PACKAGE = PACKAGE;
    const jsonData = JSON.stringify(buildConfig, null, 2);
    fs.writeFileSync("./buildConfig.json", jsonData, { encoding: "utf8" });
  }
}
function checkDependenciesChange(options) {
  const dir = __dirname + "\\";
  const cwd = options.cwd;
  const cwdKey = cwd.replace(dir, "");
  const checkKeyList = PACKAGE.checkKeyList;
  const packagePath = path.join(cwd, "package.json");
  if (!fs.existsSync(packagePath)) return;
  const packageJson = require(packagePath);
  const configJson = PACKAGE[cwdKey];
  if (!configJson) {
    updateConfigPackage(options);
    return true;
  }
  for (const checkKey of checkKeyList) {
    const obj = packageJson[checkKey];
    const obj1 = configJson[checkKey] || {};
    if (!obj) continue;
    for (const k in obj) {
      if (obj1[k] !== obj[k]) {
        updateConfigPackage(options);
        return true;
      }
    }
  }
  updateConfigPackage(options);
  return false;
}
async function npmInstall(options) {
  const dir = __dirname + "\\";
  const cwdKey = options.cwd.replace(dir, "");
  const cwd = path.join(options.cwd, "node_modules");
  if (!fs.existsSync(cwd) || checkDependenciesChange(options)) {
    console.log(`Updating dependencies, please wait...`);
    child_process.execSync("npm install", options);
  }
}
async function npmBuild(options) {
  const dir = __dirname + "\\";
  const cwd = options.cwd;
  const cwdKey = cwd.replace(dir, "");
  const hashCode = hashFolder(cwd);
  if (hashCode === HASH[cwdKey]) {
    console.log(`${cwdKey} no updates`);
    return;
  }
  HASH[cwdKey] = hashCode;
  buildConfig.HASH = HASH;
  const jsonData = JSON.stringify(buildConfig, null, 2);
  fs.writeFileSync("./buildConfig.json", jsonData, { encoding: "utf8" });
  console.log(`Packing ${cwdKey}`);
  child_process.execSync("npm run build", options);
}
async function buildPopup() {
  const cdPath = path.join(__dirname, "./popup");
  const options = {
    cwd: cdPath,
  };
  await npmInstall(options);
  await npmBuild(options);
}
async function buildPanel() {
  const cdPath = path.join(__dirname, "./panel");
  const options = {
    cwd: cdPath,
  };
  await npmInstall(options);
  await npmBuild(options);
}
async function reNameCss() {
  const cdPath = path.join(__dirname, "./panel/dist/css");
  const cssFiles = fs.readdirSync(`${cdPath}`);
  const oldName = path.join(__dirname, `./panel/dist/css/${cssFiles[0]}`);
  const newName = path.join(__dirname, `./panel/dist/css/panel.css`);
  fs.renameSync(oldName, newName);
}
async function editManifest() {
  const data = fs.readFileSync("manifest.json", "utf8");
  const jsonData = JSON.parse(data);
  const cdPath = path.join(__dirname, "./panel/dist/js");
  const jsFiles = fs.readdirSync(`${cdPath}`);
  const cdPath1 = path.join(__dirname, "./script");
  const jsFiles1 = fs.readdirSync(`${cdPath1}`);
  const web_accessible_resources = [
    ...jsFiles.map((file) => {
      return "panel/dist/js/" + file;
    }),
  ];
  web_accessible_resources.push(
    ...jsFiles1.map((file) => {
      return `script/${file}`;
    })
  );

  // console.log(web_accessible_resources)

  // 动态写入panel/css的文件
  const cdPath2 = path.join(__dirname, "./panel/dist/css");
  const cssFiles = fs.readdirSync(cdPath2).filter(file => file.endsWith('.css'));

  const cssPaths = cssFiles.map(file => `panel/dist/css/${file}`);
  if (jsonData.content_scripts && jsonData.content_scripts.length > 0) {
    jsonData.content_scripts.forEach(script => {
      script.css = [...new Set(["myStyle.css", ...cssPaths])];
    });
  }

  // 获取字体文件路径
  const fontsPath = path.join(__dirname, "./panel/dist/fonts");
  const fontFiles = fs.readdirSync(fontsPath);
  const fontPaths = fontFiles.map(file => `panel/dist/fonts/${file}`);

  web_accessible_resources.push(...fontPaths);
  jsonData.web_accessible_resources[0].resources = [
    ...new Set(web_accessible_resources),
  ];
  const modifiedJsonData = JSON.stringify(jsonData, null, 2);
  fs.writeFileSync("manifest.json", modifiedJsonData, "utf-8");
}
async function editContentScript() {
  const cdPath = path.join(__dirname, "./panel/dist/js");
  const cdPath1 = path.join(__dirname, "./script");
  const jsFiles = fs.readdirSync(`${cdPath}`);
  const jsFiles1 = fs.readdirSync(`${cdPath1}`);
  const scriptUrlList = jsFiles.map((file) => {
    return `"panel/dist/js/${file}"`;
  });
  scriptUrlList.push(
    jsFiles1.map((file) => {
      return `"script/${file}"`;
    })
  );
  const text = `
const scriptUrlList = [${scriptUrlList}];
scriptUrlList.forEach(url=>{
    const s = document.createElement('script');
    s.src = chrome.runtime.getURL(url);
    
    s.onload = function() {
        this.remove();
    };
    (document.head || document.documentElement).appendChild(s);
})`;
  fs.writeFileSync("contentScript.js", text, "utf-8");
}
async function createDist() {
  fs.removeSync("dist");
  const cpFileList = [
    "panel/dist/",
    "popup/dist/",
    "manifest.json",
    "img",
    "bg.js", "background.js",
    "contentScript.js",
    "myStyle.css",
    "script",
    "_locales",
  ];
  for (const file of cpFileList) {
    fs.copySync(file, "dist/" + file);
  }
}

async function run() {
  await buildPopup();
  await buildPanel();
  await reNameCss();
  console.log(`Generating manifest file, please wait...`);
  await editManifest();
  await editContentScript();
  console.log(`Generating dist package, please wait...`);
  await createDist();
  console.log(`Packaging completed`);
}
// 监听文件夹的哈希值变化
function watchFolder(folderPath = "./") {
  let currentHash = hashFolder(folderPath);
  console.log(`Watching folder ${folderPath}`);

  fs.watch(folderPath, { recursive: true }, (eventType, filename) => {
    if (eventType === "change") {
      const newHash = hashFolder(folderPath);
      if (newHash !== currentHash) {
        console.log(`File ${filename} has changed`);
        currentHash = newHash;
      }
    }
  });
}
run();
