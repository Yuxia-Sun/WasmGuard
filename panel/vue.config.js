const { defineConfig } = require("@vue/cli-service");
const MiniCssExtractPlugin = require('mini-css-extract-plugin');

module.exports = defineConfig({
  publicPath: process.env.NODE_ENV === 'production' ? '/dist/' : '/',
  transpileDependencies: true,
  productionSourceMap: false,
});

// module.exports = {
//   configureWebpack: {
//     optimization: {
//       splitChunks: {
//         cacheGroups: {
//           styles: {
//             name: 'panel',
//             test: /\.css$/,
//             chunks: 'all',
//             enforce: true,
//           },
//         },
//       },
//     },
//     plugins: [
//       new MiniCssExtractPlugin({
//         filename: 'css/[name].css',
//       }),
//     ],
//   },
// };
