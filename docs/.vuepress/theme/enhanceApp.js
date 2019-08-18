import globalComponents from './components/global'
import VueIntercom from 'vue-intercom'
import VueAnalytics from 'vue-analytics'

export default ({ Vue, options, router, siteData }) => {
  // register components.
  Vue.use(globalComponents)
  Vue.use(VueIntercom, { appId: 'tkos0k31' })
  Vue.use(VueAnalytics, { id: 'UA-121546945-5' })
}
