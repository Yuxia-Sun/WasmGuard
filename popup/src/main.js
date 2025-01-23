import Vue from 'vue'
import App from './App.vue'
import 'element-ui/lib/theme-chalk/index.css'; // 确保也引入了 Element UI 的 CSS
import ElementUI from 'element-ui';

Vue.config.productionTip = false

new Vue({
  render: h => h(App),
}).$mount('#app')

