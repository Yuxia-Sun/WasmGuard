const path = require("path");
const child_process = require("child_process");

const options = {
  cwd: "./",
};
child_process.execSync("npm install", options);

const fs = require("fs-extra");

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
const { PACKAGE } = buildConfig;

async function npmInstall(options) {
  const dir = __dirname + "\\";
  const cwd = options.cwd;
  const cwdKey = cwd.replace(dir, "");
  console.log(`${cwdKey}正在更新依赖，请稍等……`);
  child_process.execSync("npm install", options);
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

async function init() {
  const list = ["./popup", "./panel"];
  for (const item of list) {
    const cdPath = path.join(__dirname, item);
    const options = {
      cwd: cdPath,
    };
    await npmInstall(options);
    updateConfigPackage(options);
  }
  console.log("初始化完成");
}
init();
