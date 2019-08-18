<template>
  <div class="sidebar">
    <div class="group" v-for="sidebarGroupItem, index in sidebars" v-if="sidebarGroupItem">
      <div class="group__title">{{ sidebarGroupItem.title }}</div>
      <div class="group__body">

        <!-- render README.md in this folder -->
        <div v-if="sidebarGroupItem.to" :class="[
          'group__category',
          'category',
          {
            'category--selected': $route.fullPath === sidebarGroupItem.to,
            'category--active': $route.fullPath === sidebarGroupItem.to,
          }
        ]">
          <div class="category__label">
            <NavLink :to="sidebarGroupItem.to">{{ title(sidebarGroupItem.title || sidebarGroupOrder[index]) }}</NavLink>
          </div>
        </div>

        <!-- render headers of README.md in this folder -->
        <div v-if="sidebarGroupItem.headers && sidebarGroupItem.headers.length" v-for="header in sidebarGroupItem.headers" :class="[
          'group__category',
          'category',
          {
            'category--selected': $route.fullPath === `${sidebarGroupItem.to}#${header.slug}`,
            'category--active': $route.fullPath === `${sidebarGroupItem.to}#${header.slug}`,
          }
        ]">
          <div class="category__label">
            <NavLink :to="`${sidebarGroupItem.to}#${header.slug}`">{{ title(header.title) }}</NavLink>
          </div>
        </div>

        <!-- render other files in this folder -->
        <div v-if="sidebarGroupItem.children && sidebarGroupItem.children.length" v-for="child in sidebarGroupItem.children" :name="`${child.to}`" :class="[
          'group__category',
          'category',
          {
            'category--selected': !child.isLangNav && $route.path === child.to,
            'category--active': !child.isLangNav && $route.fullPath === child.to,
          }
        ]">
          <div class="category__label">
            <NavLink :to="child.to">{{ title(child.title) }}</NavLink>
          </div>
          <div v-if="child.headers && child.headers.length" v-for="header in child.headers" :class="[
            'category__headers',
            {
              'category--active': $route.fullPath === `${child.to}#${header.slug}`,
            }
          ]">
            <div class="category__header-item">
              <NavLink :to="`${child.to}#${header.slug}`">{{ title(header.title) }}</NavLink>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import config from '../../config'
import { title } from '../../utils'
import NavLink from '../NavLink'

export default {
  name: 'Sidebar',
  components: {
    NavLink,
  },
  props: {
    items: {
      type: Object,
      required: true,
    },
  },
  computed: {
    sidebarGroupOrder() {
      const groupOrderConfig = config.get(
        this.$site,
        'sidebarGroupOrder',
        this.$localePath
      )

      const languageSelectText = config.get(this.$site, 'selectText', this.$localePath) || 'languages'

      if (groupOrderConfig) {
        const result = groupOrderConfig.slice()

        result.unshift(languageSelectText, 'home')

        return result
      } else {
        return Object.keys(this.items)
      }
    },
    sidebars() {
      return this.sidebarGroupOrder
        .map(item => {
          return this.items[item]
        })
    },
  },
  methods: {
    title,
  },
}
</script>

<style lang="stylus">
@import '../../styles/_variables.styl'

.sidebar
  position: fixed
  top: 0
  bottom: 0
  width: 100%
  padding-top: 5rem
  overflow: auto
  background: $white

  @media (max-width: 768px)
    z-index: 100

.group
  margin-bottom: 4rem

  // &:first-child
  //   .group__title
  //     display: none

  &__title
    padding-left: 30px
    margin-bottom: 1em
    font-size: 14px
    font-weight: 300
    letter-spacing: 1.3px
    text-transform: uppercase
    color: #888
    font-weight: bold

.category
  a,
  a:hover
    color: $primary

  a
    &.router-link-exact-active
      color: black

  &__label,
  &__header-item
    height: 2em
    margin: 0.6em 0
    line-height: 2em

  &__label,
  &__headers
    border-left: 4px solid $white

  &__label
    padding-left: 26px

  &__headers
    display: none

  &--active,
  &--selected
    & ^[0]__headers
      display: block

  &--active &__label,
  &--active&__headers
      font-weight: 600
      border-color: $black

  &__header-item
    padding-left: 30px

    &::before
      margin-right: 4px
      color: #979797
      content: "-"
</style>
