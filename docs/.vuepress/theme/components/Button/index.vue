<template>
  <nav-link :class="classes" v-if="isLink" :to="to"><slot></slot></nav-link>
  <button :class="classes" type="button" v-else @click="handleClick">
    <span class="btn__loading-wrapper" v-if="loading">
      <img src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/PjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+PHN2ZyB0PSIxNTI2NTU0NTQ0OTUxIiBjbGFzcz0iaWNvbiIgc3R5bGU9IiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjE5MjYiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB3aWR0aD0iMjIwIiBoZWlnaHQ9IjIyMCI+PGRlZnM+PHN0eWxlIHR5cGU9InRleHQvY3NzIj48L3N0eWxlPjwvZGVmcz48cGF0aCBkPSJNOTIzLjg1MTU4MSAzMzguMDA3MjQ3Yy0yMC4wMTI3OTktNDcuMzEzNTg1LTQ3LjcxNzc5MS05MC4zMjQxNjgtODIuNDQ4ODA0LTEyOC4xMjgxNjkgNi44NzU1OTQtMTMuNTQ0NDgxIDEwLjc1MzkyNS0yOC44NjY0NDUgMTAuNzUzOTI1LTQ1LjA5NjA4MSAwLTU1LjA5OTkyMi00NC42NjczMTYtOTkuNzY3MjM4LTk5Ljc2NzIzOC05OS43NjcyMzgtMjkuMDUxNjYzIDAtNTUuMTk4MTYgMTIuNDIwODkxLTczLjQzMTQyOSAzMi4yMzcyMTUtNTMuMDU4NDI2LTIxLjM5MDE2OS0xMDkuMTgzNzAxLTMyLjIzNzIxNS0xNjYuOTU4NTQ2LTMyLjIzNzIxNS02MC4zMzQxMzMgMC0xMTguODczMzg4IDExLjgyMTIzMy0xNzMuOTg4NjYgMzUuMTM0MTk1LTUzLjIzMjM4OCAyMi41MTM3Ni0xMDEuMDMxMDIxIDU0Ljc0MTc2NS0xNDIuMDczNzg2IDk1Ljc4NDUzLTQxLjA0NTgzNSA0MS4wNDM3ODgtNzMuMjcyODE3IDg4Ljg0NDQ2Ny05NS43ODU1NTMgMTQyLjA3Mzc4NkM3Ni44Mzc1MDQgMzkzLjEyNjYxMiA2NS4wMTQyMjQgNDUxLjY2NTg2NyA2NS4wMTQyMjQgNTEyLjAwMTAyM2MwIDYwLjMzMzExIDExLjgyMzI4IDExOC44NzMzODggMzUuMTM0MTk1IDE3My45ODk2ODMgMjIuNTEzNzYgNTMuMjMwMzQyIDU0Ljc0MTc2NSAxMDEuMDI5OTk4IDk1Ljc4NTU1MyAxNDIuMDczNzg2IDQxLjA0MTc0MiA0MS4wNDM3ODggODguODQxMzk4IDczLjI3MDc3IDE0Mi4wNzM3ODYgOTUuNzg1NTUzIDU1LjExNjI5NSAyMy4zMTE5MzkgMTEzLjY1NTU1IDM1LjEzNDE5NSAxNzMuOTg5NjgzIDM1LjEzNDE5NXMxMTguODczMzg4LTExLjgyMjI1NyAxNzMuOTkyNzUzLTM1LjEzNDE5NWM1My4yMjkzMTktMjIuNTE0NzgzIDEwMS4wMjk5OTgtNTQuNzQyNzg4IDE0Mi4wNzI3NjMtOTUuNzg1NTUzIDQxLjA0NTgzNS00MS4wNDM3ODggNzMuMjcyODE3LTg4Ljg0MzQ0NCA5NS43ODU1NTMtMTQyLjA3Mzc4NiAyMy4zMDk4OTItNTUuMTE2Mjk1IDM1LjEzNDE5NS0xMTMuNjU2NTczIDM1LjEzNDE5NS0xNzMuOTg5NjgzQzk1OC45ODU3NzYgNDUxLjY2NTg2NyA5NDcuMTYxNDczIDM5My4xMjc2MzUgOTIzLjg1MTU4MSAzMzguMDA3MjQ3ek04ODcuODkxNTc4IDY3MC43ODMzNTNjLTIwLjU0NTk0MiA0OC41NzUzMjItNDkuOTYwODc5IDkyLjIwMzk4Mi04Ny40MzAyNTggMTI5LjY3NjQzMS0zNy40NzM0NzMgMzcuNDcwNDAzLTgxLjEwMDA4NiA2Ni44ODg0MS0xMjkuNjc5NTAxIDg3LjQzMjMwNS01MC4yODAxNSAyMS4yNjMyNzktMTAzLjcwMjg3MyAzMi4wNDc5MDQtMTU4Ljc4MzM1MyAzMi4wNDc5MDQtNTUuMDc3NDEgMC0xMDguNTAwMTMzLTEwLjc4MjU3Ny0xNTguNzg1Mzk5LTMyLjA0NzkwNC00OC41NzUzMjItMjAuNTQ1OTQyLTkyLjIwMzk4Mi00OS45NjE5MDItMTI5LjY3NjQzMS04Ny40MzIzMDUtMzcuNDcyNDUtMzcuNDcyNDUtNjYuODg4NDEtODEuMTAxMTA5LTg3LjQzMDI1OC0xMjkuNjc2NDMxLTIxLjI2NzM3My01MC4yODAxNS0zMi4wNTA5NzMtMTAzLjcwMzg5Ny0zMi4wNTA5NzMtMTU4Ljc4NDM3NnMxMC43ODM2MDEtMTA4LjUwMzIwMyAzMi4wNTA5NzMtMTU4Ljc4MzM1M2MyMC41NDE4NDktNDguNTc1MzIyIDQ5Ljk1ODgzMi05Mi4yMDYwMjggODcuNDMwMjU4LTEyOS42NzY0MzEgMzcuNDcyNDUtMzcuNDcwNDAzIDgxLjEwMTEwOS02Ni44ODczODYgMTI5LjY3NjQzMS04Ny40MzMzMjggNTAuMjg1MjY3LTIxLjI2NjM0OSAxMDMuNzA3OTktMzIuMDQ5OTUgMTU4Ljc4NTM5OS0zMi4wNDk5NSA1MC42MDA0NDUgMCA5OS43OTc5MzcgOS4xMDc0MjUgMTQ2LjQ0NzM5NyAyNy4wNzQ2MzUtMy43Njc4MTMgMTAuNTE0NDcxLTUuODI0NjU5IDIxLjg0MjQ3MS01LjgyNDY1OSAzMy42NTM0NzEgMCA1NS4wOTk5MjIgNDQuNjY3MzE2IDk5Ljc2NzIzOCA5OS43NjcyMzggOTkuNzY3MjM4IDI0LjU3MzY3NSAwIDQ3LjA2NTk0NS04Ljg4OTQ2MSA2NC40NTI5NDEtMjMuNjIwOTc3IDI5LjczNzI3OCAzMy4zNzIwNjIgNTMuNjAzODQ5IDcxLjAzNzkxNiA3MS4wNTAxOTYgMTEyLjI4NTM0MyAyMS4yNjYzNDkgNTAuMjgwMTUgMzIuMDQ5OTUgMTAzLjcwMjg3MyAzMi4wNDk5NSAxNTguNzgzMzUzUzkwOS4xNTg5NSA2MjAuNTAzMjAzIDg4Ny44OTE1NzggNjcwLjc4MzM1M3oiIHAtaWQ9IjE5MjciIGZpbGw9IiNmZmZmZmYiPjwvcGF0aD48L3N2Zz4=" alt="">
    </span>
    <slot></slot>
  </button>
</template>

<script>
import NavLink from '../NavLink'

export default {
  name: 'Button',
  components: {
    NavLink,
  },
  props: {
    to: String,
    type: {
      type: String,
      default: 'default',
    },
    size: {
      type: String,
      default: 'default',
    },
    light: {
      type: Boolean,
      default: false,
    },
    loading: {
      type: Boolean,
      default: false,
    },
  },
  computed: {
    isLink() {
      return !!this.to
    },
    classes() {
      const { light, type, size, loading } = this

      return [
        'btn',
        `btn--${light ? 'light' : type}`,
        `btn--${size}`,
        {
          'btn--loading': loading,
        },
      ]
    },
  },
  methods: {
    handleClick() {
      this.$emit('click')
    },
  },
}
</script>

<style lang="stylus">
@import '../../styles/_variables.styl'

.btn
  position: relative
  display: inline-block
  padding: 1.5em 6em
  border: 2px solid $black
  font-size: 12px
  border-radius: 4px
  transition: border 0.2s, background 0.2s, color 0.2s ease-out

  &:hover
    color: $black
    background-color: $white
    cursor: pointer

  &--default
    color: $white
    background-color: $black

  &--light
    color: $black
    border-color: $white
    background-color: $white

    &:hover
      color: $white
      background-color: $black

  &--small
    padding: 1em 4em
  &--large
    font-size: 14px

  &--loading img
    width: 1em
    height: 1em

  &__loading-wrapper
    position: absolute
    top: 0
    left: 0
    display: flex
    width: 100%
    height: 100%
    justify-content: center
    align-items: center
    background-color: $black

    img
      animation: loadingCircle 1s infinite linear

@keyframes loadingCircle {
  from {
    transform-origin: 50% 50%
    transform: rotate(0deg)
  }

  to {
    transform-origin: 50% 50%
    transform: rotate(1turn)
  }
}
</style>
