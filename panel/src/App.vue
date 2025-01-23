<template>
  <div id="chromePlugPanelApp" v-show="showPanel">
    <!-- <div id="chromePlugPanelMask" @click="closePanel()"></div> -->
    <div id="chromePlugPanel">
      <div id="chromePlugPanelMain" style="width: 100%; flex: 1; overflow:visible">
        <div style="height: 3em;" id="chromePlugPanelTitle">
          <div id="title_left">WebChecker</div>
          <div id="title_right">
            <span id="chromePlugPanelClose" @click="closePanel()">×</span>
          </div>
        </div>
        <el-table :data="tableData" border style="width: 100%;background-color:#fff; " height="410">
          <el-table-column prop="host" label="URL" width="180">
          </el-table-column>
          <el-table-column prop="originalFilename" label="Wasm Filename" width="180">
          </el-table-column>
          <el-table-column prop="filename" label="SHA-256">
          </el-table-column>
          <el-table-column prop="res" label="Alert" width="100"
            :filters="[{ text: 'benign', value: 0 }, { text: 'malicious', value: 1 }]" :filter-method="filterTag"
            filter-placement="bottom-end">
            <template slot-scope="scope">
              <el-tag :type="scope.row.res === 0 ? 'success' : 'danger'" disable-transitions>{{ scope.row.res === 0 ?
                "benign" : "malicious" }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="note" label="Note">
          </el-table-column>
        </el-table>
      </div>
    </div>
  </div>
</template>
<script>
// import tabBar from "./components/common/tabBar.vue";
export default {
  name: "chromePlugPanelApp",
  components: {
    // tabBar,
  },
  data() {
    return {
      tableData: [],
      mock: [
        {
          host: "http://localhost:8000/test.html",
          filename: "04a5d48f4d815149.wasm",
          originalFilename: "test.wasm", // 添加原始文件名
          res: 1,
          note: "模型预测1%的概率为良性软件，恶意99%的概率为恶意软件。"
        },
        {
          host: "bilibili",
          filename: "test2.wasm",
          originalFilename: "test2.wasm", // 添加原始文件名
          res: 0,
          note: "模型预测99%的概率为良性软件恶意1%的概率为恶意软件。"
        },
      ],
      // showPanel: true,
      showPanel: false,
    };
  },
  created() {
    if (chrome && chrome?.runtime && chrome?.runtime?.onMessage) {
      chrome.runtime.onMessage.addListener(
        (message, sender, sendResponse) => {
          if (message.type === "data_to_show") {
            console.log(message.data);
            // this.showPanel = true;
            this.tableData = message.data.map(item => ({
              ...item,
              originalFilename: item.originalFilename || 'Unknown' // 确保每个项都有 originalFilename
            }));
            console.log("data_to_show this.tableData: ", this.tableData);
            //if(this.tableData.length === 0 )
            //this.tableData = this.mock
          }
          return true; // 如果您在异步监听器中响应消息，确保返回true
        }
      );
    }
    if (window) {
      window.addEventListener("message", (event) => {
        // 检查event来源，确保安全性
        // if (event.origin !== "可信来源")
        //   return;
        //console.log(event)
        if (event.data.type === "data_to_show") {
          console.log("Received message from content script:", event.data);
          // this.showPanel = true;
          this.tableData = event.data.data.map(item => ({
            ...item,
            originalFilename: item.originalFilename || 'Unknown' // 确保每个项都有 originalFilename
          }));
          //if(this.tableData.length === 0 )
          //this.tableData = this.mock
          // 根据event.data进行操作
        } else {
          //this.tableData = this.mock
        }

      });
    }
  },
  mounted() {
    console.log("hello from vue plug page")
    // if (chrome && chrome?.runtime && chrome?.runtime?.sendMessage) {
    //   chrome.runtime.sendMessage({
    //     type: "vue_page_mounted"
    //   })
    // } else {
    //console.log("用原生window方法发送信息")
    window.postMessage({ type: "vue_page_mounted" }, "*");
    // }

    const keydownFn = (event) => {
      if (event.altKey && event.key === "v") {
        this.showPanel = !this.showPanel;
      }
    };
    const chromePlugPanel = document.getElementById("chromePlugPanel");
    const chromePlugPanelMask = document.getElementById("chromePlugPanelMask");
    document.addEventListener("keydown", keydownFn);
    chromePlugPanel && chromePlugPanel.addEventListener("keydown", keydownFn);
    chromePlugPanelMask &&
      chromePlugPanelMask.addEventListener("keydown", keydownFn);
    if (this.tableData.length > 0) {
      this.checkMaliciousFiles();
    }
  },
  watch: {
    // 侦听tableData的变化  
    tableData: {
      handler(newVal, oldVal) {
        // 当tableData变化时，调用checkMaliciousFiles方法  
        if (newVal.length > 0) { // 只在有内容时调用  
          this.checkMaliciousFiles();
        }
      },
      deep: true, // 如果tableData是一个对象或数组，并且你需要深度监听其内部属性的变化，可以设置为true  
      immediate: false // 如果需要在组件初始化时立即调用一次handler，可以设置为true  
    },
  },
  methods: {
    closePanel() {
      this.showPanel = false;
    },
    filterTag(value, row) {
      return row.res === value;
    },
    closeCurrentTab() {
      if (chrome && chrome?.runtime && chrome?.runtime?.sendMessage) {
        chrome.runtime.sendMessage({
          action: 'closeTab'
        })
      } else {
        //console.log("用原生window方法发送信息")
        window.postMessage({ action: 'closeTab' }, "*");
      }
    },
    checkMaliciousFiles() {
      console.log("Checking for malicious files...");
      let isMalicious = this.tableData.some(item => {
        console.log("Checking item:", item);
        return item.res >= 0.6 && item.host === window.location.host;
      });
      console.log("Is malicious:", isMalicious);
      if (isMalicious) {
        this.showPanel = true;
        this.$message({
          showClose: true,
          message: 'This webpage contains malicious Wasm files. Please stop accessing the current webpage!',
          type: 'error'
        });
      }
    },
  },
};
</script>

<style scoped>
#chromePlugPanelMask {
  position: fixed;
  background: #b6bdc4;
  opacity: 0.2;
  width: 100%;
  height: 100%;
  top: 0;
  left: 0;
}

#chromePlugPanel {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  position: fixed;
  background: #b6bdc4;
  width: 50%;
  height: 50%;
  z-index: 999;
  display: flex;
  flex-direction: column;
  left: 50%;
  /* 定位父级的50% */
  top: 50%;
  transform: translate(-50%, -50%);
  /*自己的50% */
  opacity: 1;
  background-color: #fff;
  border-radius: 16px;
}

#chromePlugPanelClose {
  text-align: center;
  font-size: 20px;
  cursor: pointer;
  padding: 0 5px;
  color: #fff;
  display: flex;
  flex-direction: row-reverse;
}

#chromePlugPanelClose:hover {
  background: red;
}

#chromePlugPanelTabBar {
  padding-right: 4em;
}

#chromePlugPanelTitle {
  display: flex;
  /*justify-content: center;/* 水平居中 */
  align-items: center;
  /* 垂直居中 */
  background-color: #5551ff;
  color: #fff;
  font-size: 20px;
  font-weight: 700;
  padding: 0 15px;
  border-radius: 16px;
  border-bottom-left-radius: 0px;
  border-bottom-right-radius: 0px;
}

#title_left {
  flex: 1;
  text-align: left;
}

#title_right {
  flex: 3;
  display: flex;
  justify-content: flex-end;
}
</style>
