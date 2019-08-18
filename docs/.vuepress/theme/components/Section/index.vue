<template>
  <section :class="classes" ref="section">
    <div class="container">
      <div class="row enhance-mode__homepage" v-if="isHomepage">
        <div class="col-md-10">
          <slot></slot>
        </div>
      </div>
      <slot v-else></slot>
    </div>
  </section>
</template>

<script>
export default {
  name: 'Section',
  props: {
    theme: {
      type: String,
      default: 'dark',
    },
    center: {
      type: Boolean,
      default: true,
    },
    enhanceMode: {
      type: Boolean,
      default: true,
    },
  },
  computed: {
    isHomepage() {
      return this.$page.frontmatter.home
    },
    classes() {
      const { theme, center, enhanceMode } = this

      return [
        'section',
        `section--${theme}`,
        {
          'section--center': center,
        },
      ]
    },
  },
  methods: {
    enhance() {
      const sectionElement = this.$refs.section
      const clientWidth = document.documentElement.clientWidth
      const originalWidth = sectionElement.offsetWidth

      sectionElement.style.width = `${clientWidth}px`
      sectionElement.style.marginLeft = `${-(clientWidth - originalWidth) /
        2}px`
    },
  },
  mounted() {
    if (this.enhanceMode) {
      this.enhance()
    }
  },
}
</script>

<style lang="stylus">
@import '../../styles/_variables.styl'

.section
  padding: 8rem 0
  margin-top: 4rem
  margin-bottom: 4rem

  &--dark
    color: $white
    background-color: $black

  &--light
    color: $black
    background-color: $white

  &--center
    text-align: center

.enhance-mode__homepage
  padding: 6rem 4rem
  justify-content: center
</style>
