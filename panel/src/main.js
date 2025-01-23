import Vue from "vue";
import App from "./App.vue";
import { Table,Button,TableColumn,Tag,Icon } from 'element-ui'; // 按需引入组件
import 'element-ui/lib/theme-chalk/index.css'; // 确保也引入了 Element UI 的 CSS
import ElementUI from 'element-ui';

Vue.use(Table);
Vue.use(Button);
Vue.use(TableColumn);
Vue.use(Tag);
Vue.use(Icon);
Vue.use(ElementUI);


Vue.config.productionTip = false;

new Vue({
  render: (h) => h(App),
}).$mount("#chromePlugPanelApp");
