<template>
  <div class="chrome-panel-tab">
    <div
      class="chrome-panel-tab-bg"
      v-for="(tab, tabIndex) in tabBarList"
      :key="'chromePanelTabItem' + tab.title + tabIndex"
      :style="getColor(tab)"
    >
      <div
        :class="{
          'chrome-panel-tab-item': true,
          'chrome-panel-tab-item-active': activeTab.title === tab.title,
        }"
        :style="getStyle(tab)"
        @click="tabChange(tab)"
      >
        {{ tab.title }}
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "tabBar",
  model: {
    prop: "activeTab",
    event: "update",
  },
  props: {
    tabList: {
      type: Array,
      default: () => {
        return [];
      },
    },
    activeTab: {
      type: Object,
      default: () => {
        return {};
      },
    },
  },
  computed: {
    tabBarList() {
      return this.tabList.map((item, index) => {
        return {
          ...item,
          index,
        };
      });
    },
    activeTabBar() {
      return this.tabBarList.find(
        (item) => item.title === this.activeTab.title
      );
    },
  },
  created() {
    this.keyDown();
  },
  methods: {
    keyDown() {
      document.addEventListener("keydown", (event) => {
        if (event.altKey && event.keyCode === 84) {
          if (!this.activeTabBar) return;
          const ind = (this.activeTabBar.index + 1) % this.tabBarList.length;
          this.tabChange(this.tabBarList[ind]);
        }
      });
    },
    tabChange(item) {
      this.$emit("update", item); // 这里更新activeTab
      this.$emit("tabChange", item);
    },
    getColor(tab) {
      let style = "";
      if (
        this.activeTabBar.index === tab.index + 1 ||
        this.activeTabBar.index === tab.index - 1
      ) {
        style += "background: #fff;";
      } else {
        style += "background: #dde3e9;";
      }
      return style;
    },
    getStyle(tab) {
      let style = "";
      if (this.activeTabBar.index == tab.index + 1) {
        style += "border-bottom-left-radius:0;";
      } else if (this.activeTabBar.index == tab.index - 1) {
        style += "border-bottom-right-radius:0;";
      } else if (
        this.activeTabBar.index < tab.index - 1 ||
        this.activeTabBar.index > tab.index + 1
      ) {
        style += "border-radius:0;";
      }
      return style;
    },
  },
};
</script>

<style scoped>
.chrome-panel-tab {
  background: #dde3e9;
  padding: 0 0.2rem;
  padding-top: 10px;
  display: flex;
  line-height: 20px;
}
.chrome-panel-tab-bg {
  flex: 1;
}
.chrome-panel-tab-item {
  cursor: pointer;
  padding: 0.2rem;
  border-bottom-left-radius: 10px;
  border-bottom-right-radius: 10px;
  background: #dde3e9;
}
.chrome-panel-tab-item-active {
  background: #fff;
  border-bottom-left-radius: 0;
  border-bottom-right-radius: 0;
  border-top-left-radius: 10px;
  border-top-right-radius: 10px;
}
.chrome-panel-tab-bg:last-child .chrome-panel-tab-item {
  border-top-right-radius: 0;
  border-bottom-right-radius: 0;
}
.chrome-panel-tab-bg:first-child .chrome-panel-tab-item {
  border-top-left-radius: 0;
  border-bottom-left-radius: 0;
}
</style>
