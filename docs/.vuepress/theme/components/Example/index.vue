<template>
  <div class="examples" type="example">
    <slot></slot>
  </div>
</template>

<script>
export default {
  name: 'Example',
}
</script>

<style lang="stylus">
@import '../../styles/_variables.styl'

.examples
  
  .custom-block.tip
    border-color: $white

    .custom-block-title
      color: $white
</style>
