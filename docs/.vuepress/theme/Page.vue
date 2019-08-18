<template>
  <div :class="pageClasses">
    <Content custom />
    <div class="content__footer-container">
      <div class="content__footer">
        <div v-if="editLink" class="edit-link">
          <a :href="editLink" target="_blank" rel="noopener noreferrer">{{ editLinkText }}</a>
          <svg viewBox="0 0 33 32" version="1.1" xmlns="http://www.w3.org/2000/svg" height="16" width="16"><g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"><g id="github" fill="#000"><path d="M16.3,0 C7.3,0 -3.55271368e-15,7.3 -3.55271368e-15,16.3 C-3.55271368e-15,23.5 4.7,29.6 11.1,31.8 C11.9,31.9 12.2,31.4 12.2,31 L12.2,28.2 C7.7,29.2 6.7,26 6.7,26 C6,24.2 5,23.7 5,23.7 C3.5,22.7 5.1,22.7 5.1,22.7 C6.7,22.8 7.6,24.4 7.6,24.4 C9.1,26.9 11.4,26.2 12.3,25.8 C12.4,24.7 12.9,24 13.3,23.6 C9.7,23.2 5.9,21.8 5.9,15.5 C5.9,13.7 6.5,12.3 7.6,11.1 C7.4,10.7 6.9,9 7.8,6.8 C7.8,6.8 9.2,6.4 12.3,8.5 C13.6,8.1 15,8 16.4,8 C17.8,8 19.2,8.2 20.5,8.5 C23.6,6.4 25,6.8 25,6.8 C25.9,9 25.3,10.7 25.2,11.1 C26.2,12.2 26.9,13.7 26.9,15.5 C26.9,21.8 23.1,23.1 19.5,23.5 C20.1,24 20.6,25 20.6,26.5 L20.6,31 C20.6,31.4 20.9,31.9 21.7,31.8 C28.2,29.6 32.8,23.5 32.8,16.3 C32.6,7.3 25.3,0 16.3,0 L16.3,0 Z" id="Shape"></path></g></g></svg>
        </div>
        <time v-if="lastUpdated" class="last-updated">
          <span class="prefix">{{ lastUpdatedText }}: </span>
          <span class="time">{{ lastUpdated }}</span>
        </time>
      </div>
    </div>
  </div>
</template>

<script>
import { isExternalLink } from './utils'

const isHeading = el => {
  const tagname = el.tagName.toLowerCase()

  return tagname === 'h1' || tagname === 'h2'
}

export default {
  data() {
    return {
      blocks: [],
    }
  },
  computed: {
    isEnchanceMode() {
      return !!this.$page.frontmatter.enhance
    },
    isBlockLayout() {
      return this.isEnchanceMode || !!this.blocks.length
    },
    pageClasses() {
      return {
        page__container: true,
        'page--block-layout': this.isBlockLayout,
      }
    },
    lastUpdated() {
      if (this.$page.lastUpdated) {
        return new Date(this.$page.lastUpdated).toLocaleString(this.$lang)
      }
    },
    lastUpdatedText() {
      if (typeof this.$site.themeConfig.lastUpdated === 'string') {
        return this.$site.themeConfig.lastUpdated
      }

      return 'Last Updated'
    },
    editLink() {
      if (this.$page.frontmatter.editLink === false) {
        return
      }

      const {
        repo,
        editLinks,
        docsDir = '',
        docsBranch = 'master',
        docsRepo = repo,
      } = this.$site.themeConfig

      let path = this.$page.path

      if (path.substr(-1) === '/') {
        path += 'README.md'
      } else {
        path += '.md'
      }

      if (docsRepo && editLinks) {
        const base = isExternalLink(docsRepo)
          ? docsRepo
          : `https://github.com/${docsRepo}`

        return (
          base.replace(/\/$/, '') +
          `/edit/${docsBranch}` +
          (docsDir ? '/' + docsDir.replace(/\/$/, '') : '') +
          path
        )
      }
    },
    editLinkText() {
      return this.$site.themeConfig.editLinkText || `Edit this page`
    },
  },
  watch: {
    $route(to, from) {
      if (to.path === from.path) return

      // Reset blocks when route changes.
      this.blocks.length = 0

      if (this.isEnchanceMode) {
        this.$nextTick(this.resolveLayout)
      }
    },
  },
  methods: {
    resolveLayout() {
      const contentContainer = this.$el.children[0]

      let html = ''

      Array.from(contentContainer.children).forEach(el => {
        if (isHeading(el)) {
          if (html) {
            html += `
                </div>
                <div class="content-block__examples">
                </div>
              </div>
            </div>
            `
          }

          html += `
            <div class="content-block">
              <div class="content-block__heading">
                ${el.outerHTML}
              </div>
              <div class="content-block__body">
                <div class="content-block__cont">
          `
        } else {
          html += el.outerHTML
        }
      })

      html += `
                </div>
                <div class="content-block__examples">
                </div>
              </div>
            </div>
      `

      contentContainer.innerHTML = html
    },
    addBlock(block) {
      this.blocks.push(block)
    },
  },
  mounted() {
    if (this.isEnchanceMode) {
      this.$nextTick(this.resolveLayout)
    }
  },
  created() {
    this.$on('addBlock', this.addBlock)
  },
}
</script>

<style lang="stylus">
@import './styles/_variables.styl'

.page__container
  min-height: 100vh
  padding: 5rem 6rem 0

  .curl__container
    text-align: center

  @media (max-width: 768px)
    padding: 80px 20px 0 20px

.content__footer
  display: flex
  justify-content: space-between
  padding: 2em 0
  font-size: 14px
  color: #999

  .edit-link
    a
      margin-right: .5em
      font-weight: 600
      color: #000
    svg
      vertical-align: middle

.page--block-layout
  .content__footer-container
    margin: 0 -3rem
    background-color: white

  .content__footer
    width: 50%;
    padding: 0 3rem 2rem
    background-color: white

.content-block
  margin: -4rem -6rem 4rem
  background-color: $black

  &:last-child
    margin-bottom: 0

  &:after
    height: 1px
    display: block
    content: ''
    width: 100%
    background-image: linear-gradient(90deg,#eaeaea 50%,#333 50%)

  &:last-child:after
    display: none

  &__heading
    width: 60%
    padding: 4rem 3rem 0
    overflow: auto
    background-color: white

  &__body
    display: flex

  &__cont,
  &__examples
    padding: 0 3rem 2rem

  &__cont
    width: 60%
    background-color: white

  &__examples
    width: 40%
    color: $white

    .btn
      margin: 2em 0

    p
      font-size: 12px

    // reset style
    blockquote
      border-left-color: $white

      p
        color: #888
</style>
