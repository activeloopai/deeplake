<template>
  <div class="theme__container" :class="pageClasses">
    <Navbar
      v-if="shouldShowNavbar"
      @toggle-sidebar="toggleSidebar"
    />
    <div class="row" v-if="shouldShowSidebar">
      <div class="col-md-2">
        <Sidebar :items="sidebarItems" :class="{ 'd-none': !isSidebarOpen }">
          <slot name="sidebar-top" slot="top" />
          <slot name="sidebar-bottom" slot="bottom" />
        </Sidebar>
      </div>
      <div class="col-md-10">
        <div class="custom__layout" v-if="$page.frontmatter.layout">
          <component :is="$page.frontmatter.layout"/>
        </div>
        <Home v-else-if="$page.frontmatter.home"/>
        <Page v-else></Page>
      </div>
    </div>
    <div v-else>
      <div class="custom__layout" v-if="$page.frontmatter.layout">
        <component :is="$page.frontmatter.layout"/>
      </div>
      <Home v-else-if="$page.frontmatter.home"/>
      <Page v-else></Page>
    </div>
  </div>
</template>

<script>
import Vue from 'vue'
import nprogress from 'nprogress'

import Navbar from './components/Navbar'
import Sidebar from './components/Sidebar'
import Home from './Home'
import Page from './Page'

import { resolveSidebarItems } from './utils'

export default {
  components: {
    Home,
    Sidebar,
    Page,
    Navbar
  },
  data() {
    return {
      isSidebarOpen: true,
    }
  },
  computed: {
    shouldShowNavbar () {
      const { themeConfig } = this.$site
      const { frontmatter } = this.$page
      if (
        frontmatter.navbar === false ||
        themeConfig.navbar === false) {
        return false
      }
      return (
        this.$title ||
        themeConfig.logo ||
        themeConfig.repo ||
        themeConfig.nav ||
        this.$themeLocaleConfig.nav
      )
    },
    sidebarItems() {
      return resolveSidebarItems(this.$page, this.$site, this.$localePath)
    },
    shouldShowSidebar() {
      const { frontmatter } = this.$page

      return (
        !frontmatter.home &&
        frontmatter.sidebar !== false &&
        Object.keys(this.sidebarItems).length
      )
    },
    pageClasses() {
      const userPageClass = this.$page.frontmatter.pageClass

      return [
        {
          'sidebar-open': this.shouldShowSidebar && this.isSidebarOpen,
          'no-sidebar': !this.shouldShowSidebar,
        },
        userPageClass,
      ]
    },
  },
  mounted() {
    // configure progress bar
    nprogress.configure({ showSpinner: false })

    if (window.innerWidth < 719) {
      this.isSidebarOpen = false
    }

    this.$router.beforeEach((to, from, next) => {
      if (to.path !== from.path && !Vue.component(to.name)) {
        nprogress.start()
      }

      this.isSidebarOpen = false

      next()
    })

    this.$router.afterEach(() => {
      nprogress.done()
    })

    this.$intercom.boot()
  },
  created() {
    if (this.$ssrContext) {
      this.$ssrContext.title = this.$title
      this.$ssrContext.lang = this.$lang
      this.$ssrContext.description = this.$page.description || this.$description
    }
  },
  methods: {
    toggleSidebar (to) {
      this.isSidebarOpen = typeof to === 'boolean' ? to : !this.isSidebarOpen
    }
  }
}
</script>

<style src="prismjs/themes/prism-tomorrow.css"></style>
<style src="./styles/theme.styl" lang="stylus"></style>
