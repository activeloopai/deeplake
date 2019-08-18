<template>
  <div class="home__container container">
    <div class="home__header">
      <h1 class="home__title">{{ title }}</h1>
      <p class="home__description">{{ description }}</p>
      <Button :to="action.link">{{ action.text }}</Button>
    </div>
    <div class="home__body row">
      <div class="home__content col-md-10">
        <Content custom />
      </div>
    </div>
    <footer class="home__footer" v-html="footer">
    </footer>
  </div>
</template>

<script>
import MarkdownIt from 'markdown-it'

import { localizePath } from './utils'
import Button from './components/Button'

const md = new MarkdownIt()

export default {
  components: {
    Button,
  },
  computed: {
    data() {
      return this.$page.frontmatter
    },
    title() {
      return this.$page.frontmatter.title || this.$title || 'Hello, World.'
    },
    description() {
      return (
        this.$page.description ||
        this.$description ||
        '📦 🎨 A api-friendly theme for VuePress.'
      )
    },
    action() {
      return {
        text: this.data.actionText,
        link: localizePath(this.data.actionLink, this.$localePath),
      }
    },
    footer() {
      const defaultFooter =
        'Open Source on [GitHub](https://github.com/sqrthree/vuepress-theme-api), Made by [@sqrthree](https://github.com/sqrthree), Power by [vuepress](https://github.com/vuejs/vuepress).'

      return md.render(this.data.footer || defaultFooter)
    },
  },
}
</script>

<style lang="stylus">
@import './styles/_variables.styl'

.home

  &__header
    margin: 10rem 0
    text-align: center

  &__title
    font-size: 46px
    font-weight: 400

  &__description
    padding: 0 20%
    margin-bottom: 5rem
    font-size: 18px
    line-height: 24px
    color: $black

  &__body
    justify-content: center

  &__content
    padding: 6rem 4rem
    border-top: 1px solid $divider-color

  &__footer
    margin: 2rem 0
    text-align: center
</style>
